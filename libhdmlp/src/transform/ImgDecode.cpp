#include "../../include/transform/ImgDecode.h"

void ImgDecode::transform(TransformPipeline* pipeline) {
    static int invaild_num = 0;// 不合法图片数量
    
    // cv::Mat temp = cv::Mat(1, (int) pipeline->src_len, CV_8UC1, pipeline->src_buffer)
    // Mat::Mat(int rows, int cols, int type, const Scalar& s)
    // 创建行数为 rows，列数为 col，类型为 type 的图像，并将所有元素初始化为值 s；
    cv::Mat temp = cv::imdecode(cv::Mat(1, (int) pipeline->src_len, CV_8UC1, pipeline->src_buffer), cv::IMREAD_COLOR);
    //从内存中读取图片，如果内存中的数据太短或者不是合法的数据就返回一个空的矩阵
    // 将temp二进制图片转换为cv::Mat

    // test
    if(temp.size().empty()){// 若图片不合法使得temp变为空矩阵
        temp = cv::Mat::zeros(cv::Size(560, 560), CV_8UC3);//就创建黑色图像替换
        invaild_num ++ ;//计数
    }
    pipeline->img = temp;
    if(invaild_num && invaild_num%10==0){//每当凑齐整十非法图片时进行输出
        printf("there has %d invalid sample.\n",invaild_num);
    }
    
}
