build_image.sh                      --              每个机器进行docker build
build_check.sh                      --              检查镜像是否build完毕
machines                            --              每一行存放worker的ip地址
mount.sh                            --              将machines中worker挂载上NoPFS和data文件夹
job_run.sh                          --              运行resnet50的脚本
build_clean                         --              清除所有容器和镜像(测试用)
job_clean                           --              清除所有任务产生的数据(测试用)
job_check.sh                        --              disabled
config-sheet                        --              记录每个job_id对应的配置情况

script                              --              存放job_run中生成的每个执行脚本
temp                                --              中间文件(可删除)

NoPFS/output/log                    --              存放各个任务的日志信息，按照配置进行存储

image : torchtest
container : torchtest
edit date : 2022/3/12