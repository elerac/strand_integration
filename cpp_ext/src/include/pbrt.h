#pragma once
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <vector>
#include "utils.h"
#include "geometry.h"
#include "camera.h"

std::string load_pbrtfile(const std::string &filename_pbrt)
{
    std::filesystem::path filename_pbrt_path(filename_pbrt);
    std::filesystem::path paretnt_path = filename_pbrt_path.parent_path();

    // Open PBRT scene file
    std::ifstream file(filename_pbrt_path);
    if (!file.is_open())
    {
        throw std::invalid_argument("Could not open file " + filename_pbrt);
    }

    std::string scene_pbrt;

    // Read every line
    std::string line;
    while (std::getline(file, line))
    {
        // Delete comments
        std::regex comment_regex("#.*");
        line = std::regex_replace(line, comment_regex, "");

        // Read every word splitted by space
        std::istringstream i_line_streaming(line);
        std::string substr;
        while (i_line_streaming >> substr)
        {
            if (substr == "Include")
            {
                i_line_streaming >> substr;                   // filename to include
                substr = substr.substr(1, substr.size() - 2); // remove double quotes
                std::filesystem::path filename_include(substr);
                std::string ext = filename_include.extension().string();
                if (ext == ".pbrt")
                {
                    scene_pbrt += load_pbrtfile((paretnt_path / filename_include).string());
                }
                continue;
            }
            scene_pbrt += substr + " ";
        }
    }

    file.close();

    return scene_pbrt;
}

std::string first_numberstring(const std::string &str, const std::size_t &pos = 0)
{
    const char *digits = "-.0123456789";
    const std::size_t n = str.find_first_of(digits, pos);
    if (n != std::string::npos)
    {
        std::size_t const m = str.find_first_not_of(digits, n);
        return str.substr(n, m != std::string::npos ? m - n : m);
    }
    return std::string();
}

std::vector<float> extract_numbers(const std::string &scene_pbrt, const std::string &attr_name, int n)
{
    std::vector<float> numbers;
    numbers.reserve(n);

    int pos = scene_pbrt.find(attr_name) + attr_name.size() + 1;
    for (int i = 0; i < n; i++)
    {
        std::string numberstring = first_numberstring(scene_pbrt, pos);
        pos += numberstring.size() + 1;
        float number = std::stof(numberstring);
        numbers.emplace_back(number);
    }

    return numbers;
}

void getCameraParametersFromPbrtFile(const std::string &filename_pbrt, cv::Matx33f &intrinsic_matrix, cv::Matx33f &rotation_matrix, cv::Vec3f &translation_vector)
{
    // Open PBRT scene file as string
    std::string scene_pbrt = load_pbrtfile(filename_pbrt);

    // Extract parameters of camera geometry from PBRT scene file
    auto numbers = extract_numbers(scene_pbrt, "LookAt", 9);
    cv::Vec3f eye(numbers.at(0), numbers.at(1), numbers.at(2));
    cv::Vec3f look(numbers.at(3), numbers.at(4), numbers.at(5));
    cv::Vec3f up(numbers.at(6), numbers.at(7), numbers.at(8));
    float fov = extract_numbers(scene_pbrt, "fov", 1).at(0);
    int width = extract_numbers(scene_pbrt, "xresolution", 1).at(0);
    int height = extract_numbers(scene_pbrt, "yresolution", 1).at(0);

    // std::cout << "Camera parameters" << std::endl;
    // std::cout << "-----------------" << std::endl;
    // std::cout << "eye: " << eye << std::endl;
    // std::cout << "look: " << look << std::endl;
    // std::cout << "up: " << up << std::endl;
    // std::cout << "fov: " << fov << std::endl;
    // std::cout << "width: " << width << std::endl;
    // std::cout << "height: " << height << std::endl;

    // Convert PBRT's camera coordinate (LookAt: `eye`, `look`, `up`) to origin point and orthonormal basis
    cv::Vec3f oc = eye;                         // Camera origin is equivalent to `eye`
    cv::Vec3f zc = cv::normalize(look - eye);   // z-axis (principal axis)
    cv::Vec3f xc = cv::normalize(zc.cross(up)); // x-axis
    cv::Vec3f yc = cv::normalize(zc.cross(xc)); // y-axis
    // cv::Vec3f xc = cv::normalize(up.cross(zc)); // x-axis
    // cv::Vec3f yc = cv::normalize(xc.cross(zc)); // y-axis

    // Extrinsic matrix (rotation matrix and translation vector)
    cv::Matx33f xyzl(xc[0], yc[0], zc[0], xc[1], yc[1], zc[1], xc[2], yc[2], zc[2]);
    rotation_matrix = xyzl.inv();
    translation_vector = -(rotation_matrix * oc);

    // Intrinsic matrix (3x3)
    float f = (std::min(width, height) * 0.5f) / std::tan(deg2rad(0.5f * fov));
    float px = width * 0.5f;
    float py = height * 0.5f;
    intrinsic_matrix = cv::Matx33f(f, 0, px, 0, f, py, 0, 0, 1);
}

Camera pbrtcamera(const std::string filename_pbrt)
{
    cv::Matx33f intrinsic_matrix, rotation_matrix;
    cv::Vec3f translation_vector;
    getCameraParametersFromPbrtFile(filename_pbrt, intrinsic_matrix, rotation_matrix, translation_vector);
    Camera camera(intrinsic_matrix, composeExtrinsicMatrix(rotation_matrix, translation_vector));
    return camera;
}