from torch.utils.tensorboard import SummaryWriter

"""
    TensorBoard主要用来对训练过程中的参数等数据做可视化，比如你可以看到训练过程中loss、梯度等数据的变化。
    1、使用之前先安装TensorBoard包：
        conda install TensorBoard
    2、编写代码，展示需要可视化的数据：
    
    3、使用命令启动TensorBoard页面;
        tensorboard --logdir=Pytorch/2-TensorBoard/logs --port=6007
"""
# SummaryWriter中的核心参数为事件文件保存位置
writer = SummaryWriter("logs")
for i in range(100):
    # add_scalar() 的三个核心参数:
    # tag (string): 相当于图表标题
    # scalar_value (float or string/blobname): 当前步的值，y轴
    # global_step (int): 当前是第几步，x轴
    writer.add_scalar("y=2x", 2 * i, i)
writer.close()
