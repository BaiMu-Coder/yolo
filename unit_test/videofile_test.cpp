#include <iostream>
#include "videofile.hpp"
#include "image_process.hpp"

int main()
{

    try
    {
        videofile vf("day--5.mp4");
        while (1)
        {
             auto frame=vf.get_next_frame();
            if (!frame || frame->empty() )
            {
                std::cout << "get_next_frame returned empty\n";
                return 1;
            }
            std::cout << "test_videofile OK, size: "
                      << frame->cols << "x" << frame->rows << std::endl;

           image_process imp(std::move(frame));
            int err=imp.image_preprocessing(640,640);
               if(err)
            {
                std::cout<<"image_preprocessing error"<<std::endl;
            }
            auto resize_image=std::move(imp._dst_image_frame);

           std::cout << "test_videofile OK, size: "
                      << resize_image->cols << "x" << resize_image->rows << std::endl;
            
       

         frame=std::move(resize_image);

            // 自适应屏幕
            // int screen_w = 640;
            // int screen_h = 1080;

            int screen_w = 1080;
            int screen_h = 640;

            double scale = std::min(
                (double)screen_w / frame->cols,
                (double)screen_h / frame->rows);

            cv::Mat show;
            cv::resize(*frame, show, cv::Size(), scale, scale); // 等比例缩放到刚好放进屏幕

            cv::namedWindow("video_test", cv::WINDOW_NORMAL);
            cv::setWindowProperty("video_test",
                                  cv::WND_PROP_FULLSCREEN,
                                  cv::WINDOW_FULLSCREEN); // 全屏显示窗口

            cv::Mat rotated;
            cv::rotate(*frame, *frame, cv::ROTATE_90_CLOCKWISE); // 顺时针 90°

            // 2. 显示这一帧
            cv::imshow("video_test", *frame);

            // 3. 等待一会儿，让窗口刷新，同时可以检测按键
            //    25~40ms 差不多就是正常视频播放速度
            int key = cv::waitKey(30);
            if (key == 'q' || key == 27)
            { // q 或 ESC 退出
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "test_videofile failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}