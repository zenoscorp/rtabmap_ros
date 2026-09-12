/*
Copyright (c) 2026, Zenos Corporation
All rights reserved. (BSD-3-Clause, see the repository root.)
*/

#include <gtest/gtest.h>
#include <rtabmap_util/MapsManager.h>
#include <rtabmap/core/Compression.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/Signature.h>
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <unistd.h>

namespace {

rtabmap::Signature gridSignature(int id, float obstacleX = 0.3f)
{
	const cv::Mat ground = (cv::Mat_<cv::Vec3f>(1, 1) << cv::Vec3f(0, 0, 0));
	const cv::Mat obstacles = (cv::Mat_<cv::Vec3f>(1, 1) << cv::Vec3f(obstacleX, 0.3f, 0.5f));
	const cv::Mat empty = (cv::Mat_<cv::Vec3f>(1, 1) << cv::Vec3f(-0.3f, 0.3f, 0));
	rtabmap::SensorData data;
	data.setId(id);
	data.setStamp(1000.0 + id);
        // The DB writer requires sensor metadata to persist the Data row.
	data.setUserData(rtabmap::compressData2(cv::Mat::ones(1, 1, CV_32FC1)));
	data.setOccupancyGrid(rtabmap::compressData2(ground), rtabmap::compressData2(obstacles),
			rtabmap::compressData2(empty), 0.1f, cv::Point3f());
	return rtabmap::Signature(id, 0, 1, data.stamp(), "",
			rtabmap::Transform(float(id), 0, 0, 0, 0, 0), rtabmap::Transform(), data);
}

class MapsManagerCacheTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		rclcpp::init(0, nullptr);
		char path[] = "/tmp/rtabmap-grid-cache-XXXXXX";
		const int fd = mkstemp(path);
		ASSERT_GE(fd, 0);
		close(fd);
		path_ = path;
		memory_.reset(new rtabmap::Memory);
		ASSERT_TRUE(memory_->init(path_, true));
		// Add persisted grids after init so they start outside working memory.
		std::unique_ptr<rtabmap::DBDriver> driver(rtabmap::DBDriver::create());
		ASSERT_TRUE(driver->openConnection(path_));
		for(int id = 1; id <= 3; ++id)
		{
			driver->asyncSave(new rtabmap::Signature(gridSignature(id)));
			driver->emptyTrashes(false);
		}
		driver->closeConnection(true);
		ASSERT_EQ(memory_->getSignature(1), nullptr);
		ASSERT_GT(memory_->getNodeData(1, false, false, false, true).gridCellSize(), 0);
		makeManager("candidate", true, candidateNode_, candidate_);
		makeManager("reference", false, referenceNode_, reference_);
	}

	void TearDown() override
	{
		candidate_.reset();
		reference_.reset();
		candidateNode_.reset();
		referenceNode_.reset();
		memory_.reset();
		if(!path_.empty()) std::remove(path_.c_str());
		rclcpp::shutdown();
	}

	void makeManager(const std::string & name, bool cleanup,
			rclcpp::Node::SharedPtr & node, std::unique_ptr<rtabmap_util::MapsManager> & manager)
	{
		rclcpp::NodeOptions options;
		options.parameter_overrides({rclcpp::Parameter("map_cleanup", cleanup)});
		node = std::make_shared<rclcpp::Node>(name, "/grid_cache_test/" + name, options);
		manager.reset(new rtabmap_util::MapsManager);
		manager->init(*node, name, true);
		manager->setParameters({{"Grid/CellSize", "0.1"}, {"GridGlobal/MinSize", "4"}});
	}

	void compareMaps()
	{
		float ax, ay, ar, bx, by, br;
		const cv::Mat a = candidate_->getGridMap(ax, ay, ar);
		const cv::Mat b = reference_->getGridMap(bx, by, br);
		ASSERT_FALSE(a.empty());
		ASSERT_EQ(a.size(), b.size());
		EXPECT_EQ(ax, bx);
		EXPECT_EQ(ay, by);
		EXPECT_EQ(ar, br);
		EXPECT_EQ(cv::norm(a, b, cv::NORM_INF), 0);
		const cv::Mat ap = candidate_->getGridProbMap(ax, ay, ar);
		const cv::Mat bp = reference_->getGridProbMap(bx, by, br);
		ASSERT_EQ(ap.size(), bp.size());
		EXPECT_EQ(ax, bx);
		EXPECT_EQ(ay, by);
		EXPECT_EQ(cv::norm(ap, bp, cv::NORM_INF), 0);
	}

	std::map<int, rtabmap::Transform> pose(int id, float correction = 0)
	{
		return {{id, rtabmap::Transform(float(id) + correction, 0, 0, 0, 0, 0)}};
	}

	std::string path_;
	std::unique_ptr<rtabmap::Memory> memory_;
	rclcpp::Node::SharedPtr candidateNode_, referenceNode_;
	std::unique_ptr<rtabmap_util::MapsManager> candidate_, reference_;
};

TEST_F(MapsManagerCacheTest, ReloadsEvictedGridsForRevisitsAndCorrections)
{
	for(const auto & step : std::vector<std::pair<int, float>>{{1, 0}, {2, 0}, {3, 0}, {1, 0}, {2, 0.15f}, {1, -0.15f}})
	{
		const auto poses = pose(step.first, step.second);
		candidate_->updateMapCaches(poses, memory_.get(), true, false);
		reference_->updateMapCaches(poses, memory_.get(), true, false);
		compareMaps();
		EXPECT_EQ(candidate_->getLocalGridCache().size(), 1u);
	}
	// map_cleanup=false remains an explicit opt-out.
	EXPECT_EQ(reference_->getLocalGridCache().size(), 3u);
}

TEST_F(MapsManagerCacheTest, RetainsWorkingMemoryGridsOutsideFilteredPoses)
{
	double loadingTime = 0;
	memory_->reactivateSignatures({1}, 1, loadingTime);
	ASSERT_NE(memory_->getSignature(1), nullptr);
	candidate_->updateMapCaches(pose(1), memory_.get(), true, false);
	candidate_->updateMapCaches(pose(2), memory_.get(), true, false);
	EXPECT_EQ(candidate_->getLocalGridCache().size(), 2u);
}

TEST_F(MapsManagerCacheTest, RetainsUnsavedAndOverriddenGrids)
{
	const std::map<int, rtabmap::Signature> unsaved{{4, gridSignature(4)}};
	const std::map<int, rtabmap::Signature> overridden{{2, gridSignature(2, 0.8f)}};
	candidate_->updateMapCaches(pose(4), memory_.get(), true, false, unsaved);
	candidate_->updateMapCaches(pose(2), memory_.get(), true, false, overridden);
	candidate_->updateMapCaches(pose(1), memory_.get(), true, false);
	EXPECT_EQ(candidate_->getLocalGridCache().size(), 3u);
	// Both payloads must still be usable without being resupplied by the caller.
	candidate_->updateMapCaches(pose(4), memory_.get(), true, false);
	EXPECT_NE(candidate_->getLocalGridCache().find(4), candidate_->getLocalGridCache().end());
	EXPECT_NE(candidate_->getLocalGridCache().find(2), candidate_->getLocalGridCache().end());
}

TEST_F(MapsManagerCacheTest, RetainsStreamedGridsWithoutMemory)
{
	for(int id = 1; id <= 3; ++id)
	{
		candidate_->updateMapCaches(pose(id), nullptr, true, false, {{id, gridSignature(id)}});
	}
	EXPECT_EQ(candidate_->getLocalGridCache().size(), 3u);
}

TEST_F(MapsManagerCacheTest, PublishesTheSameGridCloudAndOctomapAfterReload)
{
	using Grid = nav_msgs::msg::OccupancyGrid;
	using Cloud = sensor_msgs::msg::PointCloud2;
	Grid::SharedPtr candidateGrid, referenceGrid;
	Cloud::SharedPtr candidateCloud, referenceCloud;
	const auto qos = rclcpp::QoS(1).reliable().transient_local();
	auto cg = candidateNode_->create_subscription<Grid>("map", qos,
			[&](Grid::SharedPtr message) {candidateGrid = message;});
	auto rg = referenceNode_->create_subscription<Grid>("map", qos,
			[&](Grid::SharedPtr message) {referenceGrid = message;});
	auto cc = candidateNode_->create_subscription<Cloud>("cloud_map", qos,
			[&](Cloud::SharedPtr message) {candidateCloud = message;});
	auto rc = referenceNode_->create_subscription<Cloud>("cloud_map", qos,
			[&](Cloud::SharedPtr message) {referenceCloud = message;});
#if defined(WITH_OCTOMAP_MSGS) and defined(RTABMAP_OCTOMAP)
	using Octomap = octomap_msgs::msg::Octomap;
	Octomap::SharedPtr candidateOctomap, referenceOctomap;
	auto co = candidateNode_->create_subscription<Octomap>("octomap_full", qos,
			[&](Octomap::SharedPtr message) {candidateOctomap = message;});
	auto ro = referenceNode_->create_subscription<Octomap>("octomap_full", qos,
			[&](Octomap::SharedPtr message) {referenceOctomap = message;});
#endif
	rclcpp::executors::SingleThreadedExecutor executor;
	executor.add_node(candidateNode_);
	executor.add_node(referenceNode_);
	auto waitFor = [&](auto ready) {
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
		while(!ready() && std::chrono::steady_clock::now() < deadline)
		{
			executor.spin_some();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		return ready();
	};
	ASSERT_TRUE(waitFor([&] {return cg->get_publisher_count() && rg->get_publisher_count() &&
			cc->get_publisher_count() && rc->get_publisher_count();}));
	auto points = [](const Cloud & message) {
		pcl::PointCloud<pcl::PointXYZRGB> cloud;
		pcl::fromROSMsg(message, cloud);
		std::vector<std::array<float, 6>> result;
		for(const auto & p : cloud) result.push_back({p.x, p.y, p.z, float(p.r), float(p.g), float(p.b)});
		std::sort(result.begin(), result.end());
		return result;
	};
	int stamp = 10;
	for(const auto & step : std::vector<std::pair<int, float>>{{1, 0}, {2, 0}, {1, 0.15f}})
	{
		const auto poses = pose(step.first, step.second);
		candidate_->updateMapCaches(poses, memory_.get(), true, true);
		reference_->updateMapCaches(poses, memory_.get(), true, true);
		candidate_->publishMaps(poses, rclcpp::Time(stamp, 0), "map");
		reference_->publishMaps(poses, rclcpp::Time(stamp, 0), "map");
		ASSERT_TRUE(waitFor([&] {return candidateGrid && referenceGrid && candidateCloud && referenceCloud &&
				candidateGrid->header.stamp.sec == stamp && referenceGrid->header.stamp.sec == stamp &&
				candidateCloud->header.stamp.sec == stamp && referenceCloud->header.stamp.sec == stamp;}));
		EXPECT_EQ(candidateGrid->info, referenceGrid->info);
		EXPECT_EQ(candidateGrid->data, referenceGrid->data);
		EXPECT_FALSE(candidateCloud->data.empty());
		EXPECT_EQ(points(*candidateCloud), points(*referenceCloud));
#if defined(WITH_OCTOMAP_MSGS) and defined(RTABMAP_OCTOMAP)
		ASSERT_TRUE(waitFor([&] {return candidateOctomap && referenceOctomap &&
				candidateOctomap->header.stamp.sec == stamp && referenceOctomap->header.stamp.sec == stamp;}));
		EXPECT_FALSE(candidateOctomap->data.empty());
		EXPECT_EQ(candidateOctomap->data, referenceOctomap->data);
#endif
		EXPECT_EQ(candidate_->getLocalGridCache().size(), 1u);
		++stamp;
	}
}

} // namespace
