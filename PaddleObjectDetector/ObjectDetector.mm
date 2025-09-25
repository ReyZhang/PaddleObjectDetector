//
//  ObjectDetector.m
//  picodet_demo
//
//  Created by boss on 2025/7/5.
//  Copyright © 2025 reyzhang. All rights reserved.
//

#import "ObjectDetector.h"
#include "paddle_api.h"
#include "paddle_use_kernels.h"
#include "paddle_use_ops.h"
#include "timer.h"      //统计识别时长
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#import <opencv2/highgui/cap_ios.h>
#import <opencv2/highgui/ios.h>
#import <opencv2/opencv.hpp>
#include <string>
#import <sys/timeb.h>
#include <vector>


using namespace paddle::lite_api;
using namespace cv;

struct Object {
    std::string class_name;
    cv::Scalar fill_color;
    cv::Rect rec;
    float prob;
};


std::mutex mtx;
MobileConfig config;  //配置模型
std::shared_ptr<PaddlePredictor> predictor; //目标检测对象
Timer tic; //时长统计工具
long long count = 0;

double tensor_mean(const Tensor &tin) {
    auto shape = tin.shape();
    int64_t size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    double mean = 0.;
    auto ptr = tin.data<float>();
    for (int i = 0; i < size; i++) {
        mean += ptr[i];
    }
    return mean / size;
}

void neon_mean_scale(const float *din, float *dout, int size, float *mean,
                     float *scale) {
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
    
    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;
    
    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);
        
        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) / scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) / scale[2];
    }
}

std::vector<cv::Scalar> GenerateColorMap(int numOfClasses) {
    std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
    for (int i = 0; i < numOfClasses; i++) {
        int j = 0;
        int label = i;
        int R = 0, G = 0, B = 0;
        while (label) {
            R |= (((label >> 0) & 1) << (7 - j));
            G |= (((label >> 1) & 1) << (7 - j));
            B |= (((label >> 2) & 1) << (7 - j));
            j++;
            label >>= 3;
        }
        colorMap[i] = cv::Scalar(R, G, B);
    }
    return colorMap;
}

// fill tensor with mean and scale, neon speed up
void pre_process(const Mat &img_in, int width, int height, bool is_scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
    }
    // Prepare input data from image
    std::unique_ptr<Tensor> input_tensor_scale(predictor->GetInput(1));
    input_tensor_scale->Resize({1, 2});
    auto *scale_data = input_tensor_scale->mutable_data<float>();
    scale_data[0] = static_cast<float>(height) / static_cast<float>(img_in.rows);
    scale_data[1] = static_cast<float>(width) / static_cast<float>(img_in.cols);
    
    std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));
    input_tensor->Resize({1, 3, height, width});
    float means[3] = {0.485, 0.456, 0.406};
    float scales[3] = {0.229, 0.224, 0.225};
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    float scale_factor = is_scale ? 1 / 255.f : 1.f;
    im.convertTo(imgf, CV_32FC3, scale_factor);
    const float *dimg = reinterpret_cast<const float *>(imgf.data);
    float *dout = input_tensor->mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, means, scales);
}

std::vector<Object> post_process(float thresh,
                                 std::vector<std::string> class_names,
                                 std::vector<cv::Scalar> color_map,
                                 cv::Mat &image) { // NOLINT
    std::unique_ptr<const Tensor> output_tensor(predictor->GetOutput(0));
    std::unique_ptr<const Tensor> output_bbox_tensor(predictor->GetOutput(1));
    auto shape_out = output_tensor->shape();
    
    int64_t output_size = 1;
    for (auto &i : shape_out) {
        output_size *= i;
    }
    
    auto *data = output_tensor->data<float>();
    auto *bbox_num = output_bbox_tensor->data<int>();
    

    std::vector<Object> result;
    for (int i = 0; i < bbox_num[0]; i++) {

        // Class id
        auto class_id = static_cast<int>(round(data[i * 6]));
        // Confidence score
        auto score = data[1 + i * 6];
        int xmin = static_cast<int>(data[2 + i * 6]);
        int ymin = static_cast<int>(data[3 + i * 6]);
        int xmax = static_cast<int>(data[4 + i * 6]);
        int ymax = static_cast<int>(data[5 + i * 6]);
        int w = xmax - xmin;
        int h = ymax - ymin;
        
        // 过滤无效框
        if (w <= 0 || h <= 0 || score <= thresh || score > 1.f) {
            continue;
        }
        
        cv::Rect rec_clip =
        cv::Rect(xmin, ymin, w, h) & cv::Rect(0, 0, image.cols, image.rows);
        Object obj;
        obj.class_name = class_id >= 0 && class_id < class_names.size()
        ? class_names[class_id]
        : "Unknow";
        obj.prob = score;
        obj.rec = rec_clip;
        if (w > 0 && h > 0 && obj.prob <= 1 && obj.prob > thresh) {
            // 把 obj 添加到 result 里
            result.push_back(obj);
            
//            cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1);
//            std::string str_prob = std::to_string(obj.prob);
//            std::string text =
//            obj.class_name + ": " + str_prob.substr(0, str_prob.find(".") + 4);
//            int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
//            double font_scale = 1.f;
//            int thickness = 2;
//            cv::Size text_size =
//            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
//            float new_font_scale = w * 0.35 * font_scale / text_size.width;
//            text_size =
//            cv::getTextSize(text, font_face, new_font_scale, thickness, nullptr);
//            cv::Point origin;
//            origin.x = xmin + 10;
//            origin.y = ymin + text_size.height + 10;
//            cv::putText(image, text, origin, font_face, new_font_scale,
//                        cv::Scalar(0, 255, 255), thickness);
        
            
        }
    }
    
    
    return result;
}

#pragma mark - ：ObjectDetector Category
@interface ObjectDetector ()

@property(nonatomic) bool flag_init;
@property(nonatomic) bool flag_cap_photo;
@property(nonatomic) std::vector<float> scale;
@property(nonatomic) std::vector<float> mean;
@property(nonatomic) float thresh;
@property(nonatomic) long input_height;
@property(nonatomic) long input_width;
@property(nonatomic) std::vector<std::string> labels;
@property(nonatomic) std::vector<cv::Scalar> colorMap;
- (std::vector<std::string>)load_labels:(const std::string &)path;
@property(nonatomic) cv::Mat cvimg;

@property(nonatomic,strong) NSMutableArray<DetectedObjectInfo *> *resultArray;

@end




@implementation ObjectDetector




#pragma mark - ：Life Cycle

/**
 指定初始化构造器
 */
- (id)initWithModel:(ObjectDetectorModelInfo *)modelInfo
               delegate:(id<ObjectDetectorDelegate>)delegate {
    if (self = [super init]) {
        self.modelInfo = modelInfo;
        self.delegate = delegate;
    }
    return self;
}


- (void)setModelInfo:(ObjectDetectorModelInfo *)modelInfo {
    _modelInfo = modelInfo;
    
    [self loadModel];
}


- (void)loadModel {
    
    NSAssert(self.modelInfo.modelPath.length > 0, @"未配置模型文件路径");
    NSAssert([self.modelInfo.modelPath hasSuffix:@".nb"], @"提供的模型文件必须是.nb类型");
    
    NSAssert(self.modelInfo.labelsPath.length > 0, @"未配置标签文件路径");
    NSAssert([self.modelInfo.labelsPath hasSuffix:@".txt"], @"提供的标签文件必须是.txt类型");
    
    NSFileManager *manager = [NSFileManager defaultManager];
    NSAssert([manager fileExistsAtPath:self.modelInfo.modelPath], @"指定的模型文件未找到");
    
    NSAssert([manager fileExistsAtPath:self.modelInfo.labelsPath], @"指定的标签文件未找到");
    
    self.labels = [self load_labels:std::string([self.modelInfo.labelsPath UTF8String])];
    self.colorMap = GenerateColorMap((int)self.labels.size());
    self.input_width = self.modelInfo.input_width;
    self.input_height = self.modelInfo.input_height;
    self.thresh = self.modelInfo.threshold;
    
    
    MobileConfig config;
    config.set_model_from_file(std::string([self.modelInfo.modelPath UTF8String]));
    
    if (predictor == nullptr) {
        predictor = CreatePaddlePredictor<MobileConfig>(config);
    }
    
}


#pragma mark - ：Public Method

/**
 目标检测推理
 */
- (void)detectImage:(UIImage *)image {
    
    self.resultArray = @[].mutableCopy;
    
    //转成 opencv的 cvimage
    cv::Mat img_cat;
    UIImageToMat(image, img_cat);
    
    std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));
    
    input_tensor->Resize({1, 3, self.input_height, self.input_width});
    input_tensor->mutable_data<float>();
    cv::Mat img;
    
    //RGBA四通道 -》 转RGB三通道
    if (img_cat.channels() == 4) {
        cv::cvtColor(img_cat, img, CV_RGBA2RGB);
    }
    
    
    Timer pre_tic;
    pre_tic.start();
    pre_process(img, (int)self.input_height, (int)self.input_width, true);
    pre_tic.end();
    // warmup
    predictor->Run(); //预加载， 为了精确的统计时间
    tic.start();
    predictor->Run();
    tic.end();
    Timer post_tic;
    post_tic.start();
    auto rec_out = post_process(self.thresh, self.labels, self.colorMap, img);
    post_tic.end();

    
    // 遍历 C++ vector，把每一项包装成 DetectedObjectInfo
    for (const auto &obj : rec_out) {
        DetectedObjectInfo *info = [[DetectedObjectInfo alloc] init];
        // C++ std::string → NSString
        info.class_name = [NSString stringWithUTF8String:obj.class_name.c_str()];
        // float → CGFloat
        info.score      = (CGFloat)obj.prob;
        // cv::Rect → CGRect
        info.rect       = CGRectMake(obj.rec.x,
                                     obj.rec.y,
                                     obj.rec.width,
                                     obj.rec.height);
        [self.resultArray addObject:info];
    }
    
    DetectedTimeInfo *timeInfo = [[DetectedTimeInfo alloc] init];
    timeInfo.pre_process_time  = pre_tic.get_average_ms();
    timeInfo.predict_time = tic.get_average_ms();
    timeInfo.post_process_time = post_tic.get_average_ms();
    
    if ([self.delegate respondsToSelector:@selector(objectDetector:result:useTime:)]) {
        [self.delegate objectDetector:self result:self.resultArray useTime:timeInfo];
    }
    
}



/**
 获取推理器版本
 */
- (NSString *)getPredicatorVersion {
    NSAssert(![self.modelInfo.modelPath isEqualToString:@""], @"未指定目标检测模型.nb文件");
    config.set_model_from_file(std::string([self.modelInfo.modelPath UTF8String]));
    
    if (predictor == nullptr) {
        predictor = CreatePaddlePredictor<MobileConfig>(config);
    }
    
    return [NSString stringWithFormat:@"%s", predictor->GetVersion().c_str()];
}


#pragma mark - : Private Method

- (std::vector<std::string>)load_labels:(const std::string &)path {
    std::vector<std::string> labels;
    FILE *fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        return labels;
    }
    while (!feof(fp)) {
        char str[1024];
        fgets(str, 1024, fp);
        std::string str_s(str);
        labels.push_back(str);
    }
    fclose(fp);
    return labels;
}



@end
