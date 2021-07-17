import paddle.fluid as fluid
import numpy as np
import paddle

#在静态图模式下构建模型（函数）
paddle.enable_static()
#构造模型（函数）--start
a=fluid.layers.create_tensor(dtype='int64',name='a')
b=fluid.layers.create_tensor(dtype='int64',name='b')
y=fluid.layers.sum(x=[a,b])
#这是一个构造模型的过程，构造了一个y=f(a,b)=a+b的函数
x1=fluid.layers.fill_constant(shape=[3,3],value=1,dtype='int64')
x2=fluid.layers.fill_constant(shape=[3,3],value=3,dtype='int64')
x3=fluid.layers.sum(x=[x1,x2])
#这里等于构造了一个x3=x1+x2的常值函数，其中x1和x2是常数矩阵
#构造模型（函数）--end

place=fluid.CUDAPlace(0)#定义工作区，这里定义在显卡上工作
exe=fluid.executor.Executor(place)#构造执行器，一个paddle的程序只能有一个执行器
exe.run(fluid.default_startup_program())#执行器初始化

a1=np.array([[3,2,1],[2,3,1],[1,1,1]]).astype('int64')#喂给函数y=a+b中a的值
b1=np.array([[1,1,1],[1,1,1],[1,1,1]]).astype('int64')#喂给函数y=a+b中b的值

#执行器启动得到结果，feed用来喂函数自变量的值，而fetch_list用于填写想要输出的值
result=exe.run(program=fluid.default_main_program(),
                           feed={a.name:a1,b.name:b1},#以字典形式喂数据
                           fetch_list=[y,x3])#其中y,x3对应上面构造模型中的y,x3
print(result)#所以result的输出即为y，x3的计算结果，以列表方式输出