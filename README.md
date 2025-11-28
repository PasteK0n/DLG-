# DLG实验复现

### dlg指令

- `python main.py --index (25)` 攻击index(默认25)号图片

- `python main.py --image `攻击自定义图片

### 实验过程

1. 从git上下载源码

   运行`python main.py --index 25`指令，得到结果：

   ![image-20251129011405352](C:\Users\15958\AppData\Roaming\Typora\typora-user-images\image-20251129011405352.png)

   <center>图像呈现</center>

   ```bash
   2.7.1+cu128 0.22.1+cu128
   Running on cuda
   0 89.9501
   10 4.4808
   20 0.7325
   30 0.1878
   40 0.0725
   50 0.0366
   60 0.0226
   70 605.1503
   80 234.8506
   C:\Users\15958\AppData\Local\Programs\Python\Python313\Lib\site-packages\torchvision\transforms\functional.py:282: RuntimeWarning: invalid value encountered in cast
     npimg = (npimg * 255).astype(np.uint8)
   90 234.8506
   ...
   ```

   <center>输出反馈：（迭代次数，当前损失值）</center>

   可见，当60次之后，损失值变大，并保持不变，优化器崩溃。

   为了处理这个状况，我们对history_size进行修改，并且对dummy.data和ddummy.label的梯度进行剪裁：

   ```python
   optimizer = torch.optim.LBFGS([dummy_data, dummy_label], history_size=20)
   
   ...
   
   #closure()
   # 裁剪虚拟数据的梯度：防止参数变化过大
       if dummy_data.grad is not None:
           # 尝试较小的 max_norm 值，如 0.1 或 0.05
           torch.nn.utils.clip_grad_norm_([dummy_data], max_norm=0.1) 
           
   # 裁剪虚拟标签的梯度：标签更敏感，使用更小的值
       if dummy_label.grad is not None:
           torch.nn.utils.clip_grad_norm_([dummy_label], max_norm=0.01)
           
       return grad_dif
   ```

   但并没有成功。经过查询资料，原因在于模型的**初始梯度值足够小**，使得 LBFGS 优化器在迭代开始时不会采取**巨大步长**，从而使梯度变小，并使溢出不容易发生。

   ![image-20251129012927574](C:\Users\15958\AppData\Roaming\Typora\typora-user-images\image-20251129012927574.png)

   <center>采用小步长后的图片显示结果</center>

   因此，将目标对准初始化函数即可，即`net.apply(weights_init)`，不过经实验发现效果并不好。

   而梯度匹配与损失函数grad_dif有关，因此不妨从grad_dif下手。我们为grad_dif增加一个系数，使得在300次内更接近于完美解。

   ![image-20251129021318866](C:\Users\15958\AppData\Roaming\Typora\typora-user-images\image-20251129021318866.png)

   
