#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <unordered_map>

namespace fs = std::filesystem;

// 이미지 전처리 함수
std::vector<cv::Mat> preprocessing(const std::string& file_path) {
    cv::Mat img_color = cv::imread(file_path);
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> rotated_images;
    for (int angle = 0; angle < 360; angle += 120) {
        cv::Point2f center(img_gray.cols / 2.0F, img_gray.rows / 2.0F);
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated_img;
        cv::warpAffine(img_gray, rotated_img, rotation_matrix, img_gray.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
        rotated_images.push_back(rotated_img);
    }

    std::vector<cv::Mat> processed_images;
    for (const auto& rotated_img : rotated_images) {
        int height = rotated_img.rows;
        int width = rotated_img.cols;
        int start_x = (width - 224) / 2;
        int start_y = (height - 224) / 2;
        cv::Mat img_crop_gray = rotated_img(cv::Rect(start_x, start_y, 224, 224));

        // 라플라시안 필터링 적용
        cv::Mat img_lap_fil;
        cv::Laplacian(img_crop_gray, img_lap_fil, CV_32F, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::Mat img_gray_copy;
        img_crop_gray.convertTo(img_gray_copy, CV_32F); // img_crop_gray를 CV_32F 타입으로 변환하여 일치시킵니다.

        cv::Mat img_enhanced;
        img_enhanced = img_gray_copy - img_lap_fil; // 같은 타입으로 산술 연산
        img_enhanced.convertTo(img_enhanced, CV_8U);
        // cv::Mat img_enhanced;
        // img_enhanced = img_crop_gray - img_lap_fil;
        // img_enhanced.convertTo(img_enhanced, CV_8U);
        double alpha = 0.5;
        double beta = 1.0 - alpha;
        cv::Mat img_combined;
        cv::addWeighted(img_crop_gray, alpha, img_enhanced, beta, 0, img_combined);

        // CLAHE 적용
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat img_equalized;
        clahe->apply(img_combined, img_equalized);

        processed_images.push_back(img_equalized);
    }

    return processed_images;
}

// 증강 함수
std::vector<cv::Mat> augment_images(const std::vector<cv::Mat>& images) {
    std::vector<cv::Mat> augmented_images;
    for (const auto& img : images) {
        augmented_images.push_back(img);
        cv::Mat flipped_img;
        cv::flip(img, flipped_img, 1);  // Horizontal flip
        augmented_images.push_back(flipped_img);
        cv::flip(img, flipped_img, 0);  // Vertical flip
        augmented_images.push_back(flipped_img);
        cv::flip(img, flipped_img, -1); // Both horizontal and vertical flip
        augmented_images.push_back(flipped_img);
    }
    return augmented_images;
}

// 이미지 예측 함수
std::vector<float> predict_images(Ort::Session& session, const std::vector<cv::Mat>& images) {
    std::vector<float> predictions;
    Ort::AllocatorWithDefaultOptions allocator;

    // 입력 및 출력 이름 가져오기
    const char* input_name = "input";
    const char* output_name = "output";

    for (const auto& img : images) {
        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);
        cv::resize(img_rgb, img_rgb, cv::Size(224, 224));
        img_rgb.convertTo(img_rgb, CV_32F, 1.0 / 255);

        // Normalize
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std_dev = {0.229f, 0.224f, 0.225f};
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < img_rgb.rows; ++i) {
                for (int j = 0; j < img_rgb.cols; ++j) {
                    img_rgb.at<cv::Vec3f>(i, j)[c] = (img_rgb.at<cv::Vec3f>(i, j)[c] - mean[c]) / std_dev[c];
                }
            }
        }

        std::vector<int64_t> input_dims = {1, 3, 224, 224};
        std::vector<float> input_tensor_values;
        input_tensor_values.assign((float*)img_rgb.data, (float*)img_rgb.data + img_rgb.total() * img_rgb.channels());

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* output_value = output_tensors.front().GetTensorMutableData<float>();

        predictions.push_back(output_value[0]);
    }

    return predictions;
}

// 메인 함수
int main() {
// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cerr << "Usage: module.exe <image_path>" << std::endl;
//         return -1;
//     }

//     std::string image_path = argv[1];

    // ONNX 모델 파일 경로 설정
    std::wstring model_file_path = L"model.onnx";

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // ONNX Runtime 세션 생성
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, model_file_path.c_str(), session_options);

    // 테스트할 이미지 파일
    std::string test_image_file = "saved_image.jpg";

    auto processed_images = preprocessing(test_image_file);
    auto augmented_images = augment_images(processed_images);

    // 모델 예측 수행
    auto predictions = predict_images(session, augmented_images);

    // 최종 예측 계산
    std::unordered_map<float, int> freq_counter;
    for (float pred : predictions) {
        float rounded_value = round(pred * 10) / 10;
        freq_counter[rounded_value]++;
    }

    auto most_common = std::max_element(freq_counter.begin(), freq_counter.end(),
                                        [](const auto& a, const auto& b) { return a.second < b.second; });
    std::vector<float> filtered_values;
    std::copy_if(predictions.begin(), predictions.end(), std::back_inserter(filtered_values),
                 [&most_common](float pred) { return round(pred * 10) / 10 == most_common->first; });
    float final_prediction = std::accumulate(filtered_values.begin(), filtered_values.end(), 0.0f) / filtered_values.size();

    std::cout << final_prediction << std::endl;

    return 0;
}