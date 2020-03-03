#include<opencv2/opencv.hpp>
#include<math.h>

cv::Mat contrastEnhance(cv::Mat image, float weightingParam)
{
    cv::Mat intensity(image.size(), CV_32F);
    cv::Mat hsvImg, pdf;
    std::vector<cv::Mat> hsv;
    //================convert to HSV to get value channel============================

    if (image.channels() > 1) {
        image.convertTo(image, CV_32F);
        image = image / 255.;
        cv::cvtColor(image, hsvImg, CV_BGR2HSV);
        cv::split(hsvImg, hsv);
        hsv[2].copyTo(intensity);
        intensity *= 255.;
    } else {
        image.convertTo(image, CV_32F);
        image.copyTo(intensity);
    }

    //============get  pdf using calcHist============================================
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&intensity, 1, 0, cv::Mat(), pdf, 1, &histSize, &histRange, true, false);
    pdf = pdf / (intensity.cols * intensity.rows);
    //===============gamma correction by applying weighting distribution function to calculate cdf========================
    double maxPdf, minPdf;
    cv::minMaxLoc(pdf, &minPdf, &maxPdf);
    cv::Mat cdf(pdf.rows, pdf.cols, pdf.type());

    for (int i = 0; i < pdf.rows; i++) {
        pdf.at<float>(i, 0) = maxPdf * std::pow((float)(pdf.at<float>(i, 0) - minPdf) / (maxPdf - minPdf), (float)weightingParam);

        if (i == 0) {
            cdf.at<float>(i, 0) = pdf.at<float>(i, 0);
        } else {
            cdf.at<float>(i, 0) = cdf.at<float>(i - 1, 0) + pdf.at<float>(i, 0);
        }
    }

    cdf = cdf / cdf.at<float>(cdf.rows - 1, 0);
    cv::Mat result = image.clone();
    int width = result.cols;
    int height = result.rows;
    float *data = (float *)intensity.data;
    auto fillData = [&](const cv::Range & r) {
        for (size_t i = r.start; i != r.end; i++) {
            int value = data[i];
            data[i] = 255 * std::pow((float)(value / 255.), (1 - cdf.at<float>(value, 0)));
        }
    };
    cv::parallel_for_(cv::Range(0, width * height), fillData);

    //    for (int i = 0; i < result.rows; i++) {
    //        for (int j = 0; j < result.cols; j++) {
    //            int value = (int)intensity.at<float>(i, j);
    //            intensity.at<float>(i, j) = 255 * std::pow((float)(value / 255.), (float)(1 - cdf.at<float>(value, 0)));
    //        }
    //    }

    if (image.channels() > 1) {
        intensity /= 255.;
        result /= 255.;
        cv::cvtColor(result, result, CV_BGR2HSV);
        cv::split(result, hsv);
        hsv[2] = intensity;
        cv::merge(hsv, result);
        cv::cvtColor(result, result, CV_HSV2BGR);
        result *= 255.;
        result.convertTo(result, CV_8UC3);
    } else {
        intensity.copyTo(result);
        result.convertTo(result, CV_8UC1);
    }

    return result;
}
