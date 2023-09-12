#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string/erase.hpp>

#include "detect_3d_cuboid/matrix_utils.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"
#include "detect_3d_cuboid/object_3d_util.h"
#include "line_lbd/line_lbd_allclass.h"

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <unordered_set>
namespace fs = std::experimental::filesystem;

DEFINE_string(cube_slam_data_input_directory, 
              "",
              "Input directory for Cube-SLAM Data");
DEFINE_string(intrinsics_calibration_filepath, 
              "",
              "Calibration file to camera intrinsics");

const std::string kDetectionDirName = "detections";
const std::string kDetection2dFilename = "detections.csv";
const std::string kCuboidDirName = "cuboids";
const std::string kEdgeDirName = "edges";
const std::string kVisualizationDirName = "visualization";
const std::string kDelimiter = " ";

struct CubeSLAMBBox {
  uint32_t class_id_;
  Eigen::Vector4d xywh_;
  double conf_;

  CubeSLAMBBox(const uint32_t class_id, 
               const double x, const double y, 
               const double w, const double h,
               const double conf) {
    class_id_ = class_id;
    xywh_ = Eigen::Vector4d({x, y, w, h});
    conf_ = conf;
  }

  CubeSLAMBBox(const uint32_t class_id, 
               const Eigen::Vector4d &xywh,
               const double conf) 
    : class_id_(class_id), xywh_(xywh), conf_(conf) {}
};

void readImages(
    const fs::path &cube_slam_data_input_directory,
    const fs::path &image_list_filepath,
    std::map<uint64_t, cv::Mat> &frame_ids_and_images) {
  std::vector<fs::path> image_filepaths;
  std::ifstream image_list_ifile;
  image_list_ifile.open(image_list_filepath, std::ios::in);
  if (!image_list_ifile.is_open()) {
    LOG(FATAL) << "Failed to open file " << image_list_filepath;
  }
  std::string line;
  while (std::getline(image_list_ifile, line)) {
    image_filepaths.push_back(cube_slam_data_input_directory / fs::path(line));
  }
  image_list_ifile.close();
  for (const auto &image_filepath : image_filepaths) {
    uint64_t frame_id = static_cast<uint64_t>(std::stoi(image_filepath.stem()));
    cv::Mat image = cv::imread(image_filepath);
    frame_ids_and_images[frame_id] = image;
  }
}

void readDetections2d(
    const fs::path &filepath,
    std::map<uint64_t, std::vector<CubeSLAMBBox>> &frame_ids_and_bboxes) {
  std::ifstream ifile;
  ifile.open(filepath, std::ios::in);
  if (!ifile.is_open()) {
    LOG(FATAL) << "Failed to open file " << filepath;
  }
  bool first_line = true;
  std::string line, word;
  while (std::getline(ifile, line)) {
    if (first_line) {
      first_line = false;
      continue;
    }
    uint64_t frame_id;
    uint32_t class_id;
    double x, y, w, h, conf;
    std::stringstream ss(line);
    // ss >> frame_id >> class_id >> x >> y >> w >> h >> conf; 
    ss >> frame_id; 
    ss >> class_id; 
    ss >> x >> y >> w >> h >> conf; 
    frame_ids_and_bboxes[frame_id].emplace_back(class_id, x, y, w, h, conf);
  }
  ifile.close();
}

void readCameraIntrinsics(
    const fs::path &filepath, 
    std::unordered_map<uint32_t, Eigen::Matrix3d> &cam_ids_and_intrinsics) {
  std::ifstream ifile;
  ifile.open(filepath, std::ios::in);
  if (!ifile.is_open()) {
    LOG(FATAL) << "Failed to open file " << filepath;
  }
  bool first_line = true;
  std::string line, word;
  while (std::getline(ifile, line)) {
    if (first_line) {
      first_line = false;
      continue;
    }
    std::stringstream ss(line);
    std::vector<std::string> words;
    while (std::getline(ss, word, ',')) {
      boost::algorithm::erase_all(word, " ");
      words.push_back(word);
    }
    uint32_t cam_id;
    size_t img_width, img_height;
    double mat_00, mat_01, mat_02, mat_10, mat_11, mat_12, mat_20, mat_21, mat_22;
    cam_id = static_cast<uint32_t>(std::stoi(words.at(0)));
    mat_00 = static_cast<double>(std::stod(words.at(3)));
    mat_01 = static_cast<double>(std::stod(words.at(4)));
    mat_02 = static_cast<double>(std::stod(words.at(5)));
    mat_10 = static_cast<double>(std::stod(words.at(6)));
    mat_11 = static_cast<double>(std::stod(words.at(7)));
    mat_12 = static_cast<double>(std::stod(words.at(8)));
    mat_20 = static_cast<double>(std::stod(words.at(9)));
    mat_21 = static_cast<double>(std::stod(words.at(10)));
    mat_22 = static_cast<double>(std::stod(words.at(11)));
    cam_ids_and_intrinsics[cam_id] 
        << mat_00, mat_01, mat_02,
           mat_10, mat_11, mat_12,
           mat_20, mat_21, mat_22;
  }
  ifile.close();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = true;

  if (FLAGS_cube_slam_data_input_directory.empty()) {
    LOG(ERROR) << "No Cube-SLAM input directory provided";
    exit(1);
  }

  if (FLAGS_intrinsics_calibration_filepath.empty()) {
    LOG(ERROR) << "No intrinsics calibration file provided";
    exit(1);
  }

  size_t nCuboids = 0;
  std::unordered_map<uint32_t, uint32_t> class_ids_and_ncuboids;
  std::vector<uint32_t> cuboid_frameids;

  std::unordered_map<uint32_t, Eigen::Matrix3d> cam_ids_and_intrinsics;
  readCameraIntrinsics(FLAGS_intrinsics_calibration_filepath, cam_ids_and_intrinsics);

  // const std::unordered_set<uint32_t> camera_ids = {1, 2};
  const std::unordered_set<uint32_t> camera_ids = {1};
  std::unordered_map<uint32_t, fs::path> cam_ids_and_image_directories;
  std::unordered_map<uint32_t, fs::path> cam_ids_and_detection_2d_paths;
  std::unordered_map<uint32_t, fs::path> cam_ids_and_cuboid_dirs;
  std::unordered_map<uint32_t, fs::path> cam_ids_and_edges_dirs;
  std::unordered_map<uint32_t, fs::path> cam_ids_and_viz_dirs;
  for (const uint32_t cam_id : camera_ids) {
    cam_ids_and_image_directories[cam_id] = 
        fs::path(FLAGS_cube_slam_data_input_directory) / std::to_string(cam_id);
    if (!fs::exists(cam_ids_and_image_directories.at(cam_id))) {
      LOG(FATAL) << "Cannot find directory " << cam_ids_and_image_directories.at(cam_id) 
          << "! You need to ensure you ran the cube_slam_data_generator script before calling this executable.";
    }
    cam_ids_and_detection_2d_paths[cam_id] 
        = cam_ids_and_image_directories.at(cam_id) / kDetectionDirName / kDetection2dFilename;
    if (!fs::exists(cam_ids_and_detection_2d_paths.at(cam_id))) {
      LOG(FATAL) << "Cannot find file " << cam_ids_and_detection_2d_paths.at(cam_id) 
          << "! You need to ensure you ran the cube_slam_data_generator script before calling this executable.";
    }
    cam_ids_and_cuboid_dirs[cam_id]
        = cam_ids_and_image_directories.at(cam_id) / kCuboidDirName;
    if (!fs::is_directory(cam_ids_and_cuboid_dirs.at(cam_id)) 
        || !fs::exists(cam_ids_and_cuboid_dirs.at(cam_id))) {
      if (!fs::create_directory(cam_ids_and_cuboid_dirs.at(cam_id))) {
        LOG(FATAL) << "failed to create directory " << cam_ids_and_cuboid_dirs.at(cam_id);
      }
    }
    cam_ids_and_edges_dirs[cam_id]
        = cam_ids_and_image_directories.at(cam_id) / kEdgeDirName;
    if (!fs::is_directory(cam_ids_and_edges_dirs.at(cam_id)) 
        || !fs::exists(cam_ids_and_edges_dirs.at(cam_id))) {
      if (!fs::create_directory(cam_ids_and_edges_dirs.at(cam_id))) {
        LOG(FATAL) << "failed to create directory " << cam_ids_and_edges_dirs.at(cam_id);
      }
    }
    cam_ids_and_viz_dirs[cam_id]
        = cam_ids_and_image_directories.at(cam_id) / kVisualizationDirName;
    if (!fs::is_directory(cam_ids_and_viz_dirs.at(cam_id)) 
        || !fs::exists(cam_ids_and_viz_dirs.at(cam_id))) {
      if (!fs::create_directory(cam_ids_and_viz_dirs.at(cam_id))) {
        LOG(FATAL) << "failed to create directory " << cam_ids_and_viz_dirs.at(cam_id);
      }
    }
  }

  for (const uint32_t cam_id : camera_ids) {
    Eigen::Matrix3d calib = cam_ids_and_intrinsics.at(cam_id);

    // detect all frames' cuboids.
    detect_3d_cuboid detect_cuboid_obj;
    // TODO (Taijing) Turn them off after debugging
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = false;
    detect_cuboid_obj.print_details = true;  // false  true
    detect_cuboid_obj.set_calibration(calib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.nominal_skew_ratio = 2;
    detect_cuboid_obj.whether_save_final_images = true;

    line_lbd_detect line_lbd_obj;
    line_lbd_obj.use_LSD = true;
    // TODO (Taijing): Maybe we need to tune this..?
    line_lbd_obj.line_length_thres = 15;  // remove short edges

    fs::path image_list_filepath 
        = fs::path(FLAGS_cube_slam_data_input_directory) / std::to_string(cam_id) / ("cam_" + std::to_string(cam_id) + "_images.txt");
    std::map<uint64_t, cv::Mat> frame_ids_and_images;
    readImages(FLAGS_cube_slam_data_input_directory, image_list_filepath, frame_ids_and_images);
    LOG(INFO) << "Finish reading images";
    std::map<uint64_t, std::vector<CubeSLAMBBox>> frame_ids_and_bboxes;
    readDetections2d(cam_ids_and_detection_2d_paths.at(cam_id), frame_ids_and_bboxes);
    LOG(INFO) << "Finish reading 2d detections";

    for (const auto &frame_id_and_image : frame_ids_and_images) {
      const auto &frame_id = frame_id_and_image.first;
      const auto &image = frame_id_and_image.second;

      fs::path filepath_cuboid = cam_ids_and_cuboid_dirs.at(cam_id) 
          / (std::to_string(frame_id)+".csv");
      std::ofstream ofile_cuboid;
      ofile_cuboid.open(filepath_cuboid, std::ios::trunc);
      if (!ofile_cuboid.is_open()) {
        LOG(FATAL) << "Failed to open file " << filepath_cuboid;
      }
      LOG(INFO) << "Saving cuboids to " << filepath_cuboid;
      if (frame_ids_and_bboxes.find(frame_id) == frame_ids_and_bboxes.end()) {
        LOG(INFO) << "No object detected for frame " << frame_id;
        continue;
      }

      cv::Mat all_lines_mat;
      line_lbd_obj.detect_filter_lines(image, all_lines_mat);
      Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
      for (int rr=0;rr<all_lines_mat.rows;rr++) {
        for (int cc=0;cc<4;cc++) {
          all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
        }
      }

      std::vector<Eigen::Vector4d> all_inobj_edges;
      std::vector<cuboid*> all_cuboids;
      for (const auto &bbox : frame_ids_and_bboxes.at(frame_id)) {
        
        // TODO Using a dummy variable now. Need to double-check if this is correct;
        Eigen::Matrix4d transToWolrd = Eigen::Matrix4d::Identity();
        transToWolrd.block(0,0,3,3) <<  1.0000000,  0.0000000,  0.0000000,
                                        0.0000000,  0.9982016, -0.0599460,
                                        0.0000000,  0.0599460,  0.9982016;
        // std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
        detect_cuboid_obj.whether_sample_cam_roll_pitch = false;

        // first frame doesn't need to sample cam pose. could also sample. doesn't matter much
        // detect_cuboid_obj.whether_sample_cam_roll_pitch = (frame_id!=0); 
        //sample around first frame's pose
        // if (detect_cuboid_obj.whether_sample_cam_roll_pitch) { 
        //   transToWolrd = fixed_init_cam_pose_Twc.to_homogeneous_matrix();
        // } {
        //   transToWolrd = curr_cam_pose_Twc.to_homogeneous_matrix();
        // }

        std::vector<ObjectSet> frames_cuboids; // each 2d bbox generates an ObjectSet, which is vector of sorted proposals
        std::cout << "class_id: " << bbox.class_id_ << std::endl;
        fs::path edge_savepath = cam_ids_and_edges_dirs.at(cam_id) / (std::to_string(frame_id) + ".csv");
        auto inobj_edges = detect_cuboid_obj.detect_cuboid(image, transToWolrd, bbox.xywh_.transpose(), all_lines_raw, frames_cuboids, edge_savepath);
       
	      bool has_detected_cuboid = frames_cuboids.size()>0 && frames_cuboids[0].size()>0;
        if (has_detected_cuboid) {
          LOG(INFO) << "Cuboid detected for frame " << frame_id;
          cuboid* detected_cube = frames_cuboids[0][0];
          ofile_cuboid << detected_cube->pos(0) << kDelimiter 
                      << detected_cube->pos(1) << kDelimiter
                      << detected_cube->pos(2) << kDelimiter
                      << detected_cube->rotY << kDelimiter
                      << detected_cube->scale(0) << kDelimiter
                      << detected_cube->scale(1) << kDelimiter
                      << detected_cube->scale(2) << kDelimiter
                      << bbox.xywh_(0) << kDelimiter
                      << bbox.xywh_(1) << kDelimiter
                      << bbox.xywh_(2) << kDelimiter
                      << bbox.xywh_(3) << kDelimiter
                      << bbox.class_id_ << std::endl;
          cuboid_frameids.push_back(frame_id);
          ++nCuboids;
          if (class_ids_and_ncuboids.find(bbox.class_id_) == class_ids_and_ncuboids.end()) {
            class_ids_and_ncuboids[bbox.class_id_] = 0;
          }
          ++class_ids_and_ncuboids[bbox.class_id_];
          all_cuboids.emplace_back(detected_cube);
        }
      }
      
      {
        cv::Mat vizImage = image.clone();
        for (int i = 0; i < all_lines_raw.rows(); ++i) {
          cv::line(vizImage, cv::Point(all_lines_raw(i,0), all_lines_raw(i,1)), cv::Point(all_lines_raw(i,2), all_lines_raw(i,3)), 
              cv::Scalar(180, 130, 70), 2, 8, 0);
        }
        // for (const auto edge : all_inobj_edges) {
        //   cv::line(vizImage, cv::Point(edge(0), edge(1)), cv::Point(edge(2), edge(3)), 
        //       cv::Scalar(255, 0, 0), 2, 8, 0);
        // }
        for (const auto &bbox : frame_ids_and_bboxes.at(frame_id)) {
          cv::rectangle(vizImage, cv::Point(bbox.xywh_(0), bbox.xywh_(1)), cv::Point(bbox.xywh_(0) + bbox.xywh_(2), bbox.xywh_(1) + bbox.xywh_(3)),
              cv::Scalar(0, 165, 255), 2, cv::LINE_8);
        }
        for (const auto cuboid : all_cuboids) {
          plot_image_with_cuboid(vizImage, cuboid);
        }
        fs::path vizSavepath = cam_ids_and_viz_dirs.at(cam_id) / (std::to_string(frame_id) + ".png");
        cv::imwrite(vizSavepath, vizImage);
      }
      

      std::cout << "nCuboids: " << nCuboids << std::endl;
      for (const auto class_id_and_num : class_ids_and_ncuboids) {
        std::cout << class_id_and_num.first << ": " << class_id_and_num.second << std::endl;
      }
      std::cout << "cuboid_frameids size: " << cuboid_frameids.size() << std::endl;
      std::cout << "frame ids: ";
      for (const auto frame_id : cuboid_frameids) {
        std::cout << frame_id << " ";
      }
      std::cout << std::endl;
      std::cout << "===============================" << std::endl;
      ofile_cuboid.close();
    }
  }

  return 0;
}