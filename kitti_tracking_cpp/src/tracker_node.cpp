#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "tracker.hpp"

namespace kitti_tracking {

const std::vector<cv::Scalar> PALETTE = {
    cv::Scalar(56, 56, 255),   cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
    cv::Scalar(29, 178, 255),  cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),  cv::Scalar(134, 219, 61),  cv::Scalar(65, 147, 26),
    cv::Scalar(211, 181, 35),  cv::Scalar(243, 123, 18),  cv::Scalar(237, 71, 19)
};

class CppTrackerNode : public rclcpp::Node {
public:
    CppTrackerNode() : Node("cpp_tracker_node") {
        this->declare_parameter("max_age", 5);
        this->declare_parameter("min_hits", 2);
        this->declare_parameter("iou_threshold", 0.3);
        this->declare_parameter("class_aware", false);

        int max_age = this->get_parameter("max_age").as_int();
        int min_hits = this->get_parameter("min_hits").as_int();
        double iou_thr = this->get_parameter("iou_threshold").as_double();
        bool class_aware = this->get_parameter("class_aware").as_bool();

        tracker_ = std::make_unique<Tracker>(max_age, min_hits, iou_thr, class_aware);

        // Subscriptions
        sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&CppTrackerNode::image_callback, this, std::placeholders::_1)
        );

        sub_dets_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "/detections", 10,
            std::bind(&CppTrackerNode::detections_callback, this, std::placeholders::_1)
        );

        // Publishers
        pub_tracks_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/tracks", 10);
        pub_image_tracked_ = this->create_publisher<sensor_msgs::msg::Image>("/image_tracked", 10);

        RCLCPP_INFO(this->get_logger(), "C++ High-Performance Tracker Node initialized.");
    }

private:
    std::unique_ptr<Tracker> tracker_;
    cv::Mat latest_image_;
    std::mutex image_mutex_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_dets_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_tracks_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_tracked_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            std::lock_guard<std::mutex> lock(image_mutex_);
            latest_image_ = cv_ptr->image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void detections_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
        std::vector<Detection> dets;
        for (const auto& d : msg->detections) {
            double cx = d.bbox.center.position.x;
            double cy = d.bbox.center.position.y;
            double w = d.bbox.size_x;
            double h = d.bbox.size_y;

            double x1 = cx - w / 2.0;
            double y1 = cy - h / 2.0;
            double x2 = cx + w / 2.0;
            double y2 = cy + h / 2.0;

            int cls_id = 0;
            double score = 1.0;
            if (!d.results.empty()) {
                cls_id = std::stoi(d.results[0].hypothesis.class_id);
                score = d.results[0].hypothesis.score;
            }

            dets.push_back({{x1, y1, x2, y2}, score, cls_id});
        }

        // Run Kalman-Hungarian tracking
        auto active_tracks = tracker_->step(dets);

        // Publish /tracks
        vision_msgs::msg::Detection2DArray tracks_msg;
        tracks_msg.header = msg->header;

        for (const auto& trk : active_tracks) {
            vision_msgs::msg::Detection2D d;
            d.header = msg->header;
            d.id = std::to_string(trk.id);

            double w = trk.bbox[2] - trk.bbox[0];
            double h = trk.bbox[3] - trk.bbox[1];
            d.bbox.center.position.x = trk.bbox[0] + w / 2.0;
            d.bbox.center.position.y = trk.bbox[1] + h / 2.0;
            d.bbox.size_x = w;
            d.bbox.size_y = h;

            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(trk.class_id);
            hyp.hypothesis.score = trk.score;
            d.results.push_back(hyp);

            tracks_msg.detections.push_back(d);
        }
        pub_tracks_->publish(tracks_msg);

        // Annotate & Publish /image_tracked
        cv::Mat vis_img;
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            if (!latest_image_.empty()) {
                vis_img = latest_image_.clone();
            }
        }

        if (!vis_img.empty()) {
            for (const auto& trk : active_tracks) {
                int x1 = static_cast<int>(trk.bbox[0]);
                int y1 = static_cast<int>(trk.bbox[1]);
                int x2 = static_cast<int>(trk.bbox[2]);
                int y2 = static_cast<int>(trk.bbox[3]);

                cv::Scalar color = PALETTE[trk.id % PALETTE.size()];
                cv::rectangle(vis_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                std::string label = "ID:" + std::to_string(trk.id) + " Cls:" + std::to_string(trk.class_id);
                cv::putText(vis_img, label, cv::Point(x1, std::max(20, y1 - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }

            auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", vis_img).toImageMsg();
            pub_image_tracked_->publish(*out_msg);
        }
    }
};

} // namespace kitti_tracking

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<kitti_tracking::CppTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
