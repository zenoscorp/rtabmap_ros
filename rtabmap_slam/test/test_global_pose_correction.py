"""Exercise stored priors through the real CoreWrapper scan callback.

At 1 m/s or 1 rad/s, a 0.2 s offset makes a reversed correction differ
by 0.4 m or 0.4 rad, far above float32 database storage noise.
"""

import math
import os
import sqlite3
import struct
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped

from launch import LaunchDescription

from launch_ros.actions import Node

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers

import pytest

import rclpy
from rclpy.node import Node as RosNode
from rclpy.time import Time

from rtabmap_msgs.msg import Info

from sensor_msgs.msg import LaserScan

from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

_OFFSETS = (-0.2, 0.0, 0.2)
_MOTIONS = ("forward", "reverse", "rotating")
_SITE = (10.0, 20.0, 0.4)
_CASES = (
    [("straight_x10_to_x11", -0.2)]
    + [(motion, offset) for motion in _MOTIONS for offset in _OFFSETS]
    + [("sensor_offset_turn", -0.2), ("missing_motion_tf", 1.0)]
)


def compose_se2(a, b):
    """Compose planar transforms, applying b then a."""
    x, y, yaw = a
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], yaw + b[2]


def _pose(motion: str, case: int, dt: float) -> tuple[float, float, float]:
    if motion == "straight_x10_to_x11":
        return 11.0 + 5.0 * dt, 0.0, 0.0
    if motion == "sensor_offset_turn":
        return 4.0, 3.0, math.pi / 2 + dt * math.pi / 0.4
    if motion == "missing_motion_tf":
        return 6.0, 2.0, 0.0
    if motion == "rotating":
        angle = 0.3 + dt
        return case + 2.0 * math.cos(angle), 2.0 * math.sin(angle), angle
    return case + (dt if motion == "forward" else -dt), 0.0, 0.3


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description() -> Any:
    """Launch the resolved rtabmap_slam executable with only scan and TF inputs."""
    directory = tempfile.TemporaryDirectory(prefix="rtabmap-prior-compensation-")
    database = Path(directory.name) / "priors.db"
    expected: list[tuple[float, str, tuple[float, float, float]]] = []
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="prior_test",
        output="screen",
        parameters=[
            {
                "database_path": str(database),
                "frame_id": "base",
                "odom_frame_id": "odom",
                "map_frame_id": "map",
                "subscribe_depth": False,
                "subscribe_rgb": False,
                "subscribe_scan": True,
                "publish_tf": False,
                "wait_for_transform": 0.2,
                "Rtabmap/DetectionRate": "0",
                "RGBD/LinearUpdate": "0",
                "RGBD/AngularUpdate": "0",
                "RGBD/ProximityBySpace": "false",
                "RGBD/ProximityByTime": "false",
                "Reg/Strategy": "1",
                "Mem/STMSize": "100",
                "Mem/RehearsalSimilarity": "1.0",
                "Optimizer/PriorsIgnored": "false",
                "Optimizer/GravitySigma": "0",
            }
        ],
    )
    return LaunchDescription([rtabmap, launch_testing.actions.ReadyToTest()]), {
        "database": database,
        "expected": expected,
        "directory": directory,
    }


class TestPriorInputs(unittest.TestCase):
    """Send one prior before each scan, with TF bracketing both timestamps."""

    def test_publish_cases(self, expected: list[Any]) -> None:
        # The isolated runner assigns a DDS domain to this test; join it, or
        # use the default domain when no runner set one.
        domain = os.environ.get("ROS_DOMAIN_ID")
        rclpy.init(domain_id=int(domain) if domain else None)
        node = RosNode("prior_test_inputs")
        try:
            self._publish_cases(node, expected)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def _publish_cases(self, node: RosNode, expected: list[Any]) -> None:
        scan_pub = node.create_publisher(LaserScan, "scan", 10)
        prior_pub = node.create_publisher(PoseWithCovarianceStamped, "global_pose", 10)
        tf = TransformBroadcaster(node)
        static_tf = StaticTransformBroadcaster(node)
        sensor = TransformStamped()
        sensor.header.stamp = node.get_clock().now().to_msg()
        sensor.header.frame_id = "base"
        sensor.child_frame_id = "sensor"
        sensor.transform.translation.x = 0.5
        sensor.transform.rotation.w = 1.0
        static_tf.sendTransform(sensor)
        received: list[float] = []

        def on_info(msg: Info) -> None:
            received.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        node.create_subscription(Info, "info", on_info, 10)

        def wait(predicate: Any, seconds: float = 15.0) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and not predicate():
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertTrue(predicate(), "RTAB-Map did not consume the test input")

        wait(
            lambda: scan_pub.get_subscription_count() > 0
            and prior_pub.get_subscription_count() > 0
            and node.count_subscribers("/tf") > 0
        )
        origin = node.get_clock().now().nanoseconds * 1e-9
        for case, (motion, offset) in enumerate(_CASES):
            stamp = origin + 2.0 * case
            transforms = []
            for dt in (-0.3, -0.2, 0.0, 0.2, 0.3):
                x, y, yaw = _pose(motion, case, dt)
                transform = TransformStamped()
                transform.header.stamp = Time(seconds=stamp + dt).to_msg()
                transform.header.frame_id = "odom"
                transform.child_frame_id = "base"
                transform.transform.translation.x = x
                transform.transform.translation.y = y
                transform.transform.rotation.z = math.sin(yaw / 2)
                transform.transform.rotation.w = math.cos(yaw / 2)
                transforms.append(transform)
            tf.sendTransform(transforms)
            prior = PoseWithCovarianceStamped()
            prior.header.stamp = Time(seconds=stamp + offset).to_msg()
            # CoreWrapper uses this frame to find the sensor-to-base extrinsic.
            prior.header.frame_id = "base"
            site = _SITE if motion in _MOTIONS else (0.0, 0.0, 0.0)
            prior_pose = compose_se2(site, _pose(motion, case, offset))
            truth = compose_se2(site, _pose(motion, case, 0.0))
            if motion == "sensor_offset_turn":
                prior.header.frame_id = "sensor"
                prior_pose = compose_se2(prior_pose, (0.5, 0.0, 0.0))
            elif motion == "missing_motion_tf":
                # The future prior is outside all published dynamic TF. The
                # fallback must preserve this deliberately different base pose.
                prior_pose = (8.0, 5.0, 0.6)
                truth = prior_pose
            x, y, yaw = prior_pose
            prior.pose.pose.position.x = x
            prior.pose.pose.position.y = y
            prior.pose.pose.orientation.z = math.sin(yaw / 2)
            prior.pose.pose.orientation.w = math.cos(yaw / 2)
            for index in (0, 7, 14, 21, 28, 35):
                prior.pose.covariance[index] = 0.01
            prior_pub.publish(prior)
            # Allow the independent TF and global-pose callbacks to drain before
            # the scan callback consumes them; the stored link proves ingestion.
            until = time.monotonic() + 0.3
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=0.02)
            scan = LaserScan()
            scan.header.stamp = Time(seconds=stamp).to_msg()
            scan.header.frame_id = "base"
            scan.angle_min = -1.5
            scan.angle_max = 1.5
            scan.angle_increment = 0.01
            scan.range_min = 0.1
            scan.range_max = 20.0
            scan.ranges = [5.0 + 0.2 * math.sin(i * 0.1) for i in range(301)]
            scan_pub.publish(scan)
            wait(lambda: any(abs(t - stamp) < 1e-5 for t in received))
            expected.append(
                (
                    stamp,
                    f"{motion}/{offset:+.1f}s",
                    truth,
                )
            )


@launch_testing.post_shutdown_test()
class TestStoredPriors(unittest.TestCase):
    """Read actual pose-prior links after RTAB-Map flushes its database."""

    def test_exit_code(self, proc_info) -> None:
        launch_testing.asserts.assertExitCodes(proc_info)

    def test_compensation(
        self, database: Path, expected: list[Any], directory: Any
    ) -> None:
        try:
            self.assertEqual(len(expected), len(_CASES))
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT n.stamp, l.transform FROM Node n "
                    "JOIN Link l ON n.id=l.from_id WHERE l.type=7"
                ).fetchall()
            self.assertEqual(len(rows), len(expected), "Every scan must store a prior")
            for stamp, label, truth in expected:
                with self.subTest(case=label):
                    matches = [blob for t, blob in rows if abs(t - stamp) < 1e-5]
                    self.assertEqual(len(matches), 1)
                    matrix = struct.unpack("<12f", matches[0])
                    actual = matrix[3], matrix[7], math.atan2(matrix[4], matrix[0])
                    errors = [a - b for a, b in zip(actual, truth)]
                    errors[2] = math.atan2(math.sin(errors[2]), math.cos(errors[2]))
                    print(f"{label}: stored={actual}, expected={truth}, error={errors}")
                    # 20 micrometres / microradians tolerates float32 database
                    # storage and timestamp rounding, with >10,000x sign margin.
                    for error in errors:
                        self.assertLess(abs(error), 2e-5, label)
        finally:
            directory.cleanup()
