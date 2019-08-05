# mnist

## usage

### 解压数据集
  在本目录下解压 `dat.7z` 
  - 如果没有安装过 7z，先安装
    ```bash
    sudo apt-get install p7zip   # Ubuntu and alike
    # sudo yum install p7zip     # Centos and alike
    ```

  - 然后执行解压
    ```bash
    7z x dat.7z
    # 或：
    # 7za x dat.7z # 这个一般是 Centos 下的命令
    # 7zr x dat.7z # 或这个
    ```
    得到 `dat/` 文件夹，里面包含四个未解压的文件和两个文件夹分别包含训练数据集和测试数据集。

### 预处理
  - 安装库
    ```bash
    pip3 install -r requirements.txt
    ```
  - 进入 `exe/` 文件夹，执行预处理，读取刚才解压的数据，节省以后训练/测试的时间
    ```bash
    python3 preprocess # 可能会很慢
    ```
    这将在 `mnist/` 文件夹下生成一个 `dmp/` 文件夹，包含训练数据集(`size: 51239504=49M`)和测试数据集(`size: 8539404=8.2M`)。
    - 如果目标路径已经存在同名文件(`training_set.pickle` 和 `testing_set.pickle`)，则直接跳过。
      要覆盖路径上的文件，使用
      ```bash
      python3 preprocess -f
      ```
  - 以后训练出的权重也将备份在这一文件夹下

### 训练
  - 进入 `exe/` 文件夹进行训练
    ```bash
    python3 train.py
    ```
    如果 `mnist/dmp/` 文件夹下存在之前备份过的权重文件，则 [`train.py`](https://github.com/Blurgyy/spring2019/blob/master/DIP/mnist/exe/train.py) 将在该权重的基础上继续进行训练。
    - 要指定权重文件，使用
      ```bash
      python3 train.py --continue $WEIGHTS_PATH # 此处命令行参数为权重文件路径
      # 如
      # python3 train.py --continue ../dmp/w.pickle
      ```
    - 或者强制重新训练（即不读取任何预先训练的权重文件）
      ```bash
      python3 train.py --retrain
      ```
    - 要指定训练迭代次数，使用
      ```bash
      python3 train.py --epoch <n> # 命令行参数接受一个正整数，默认迭代次数为 100 次
      # 如：要指定迭代次数为 5，则使用
      # python3 train.py --epoch 5
      ```
    - 以上参数可以结合使用。但当参数 `--retrain` 激活时， `--continue` 被忽略。
    - 学习率随训练迭代次数降低，默认函数是一个类 `sigmoid` 函数（在我的电脑上这一函数的训练效果和测试效果都是最好的），要修改这个函数，*注释*/*取消注释* [这里](https://github.com/Blurgyy/spring2019/blob/4cb6641e71544327e10c7aba56560ed0ce86a132/DIP/mnist/exe/train.py#L149)。
    - 训练过程中，每训练一个 `epoch` ，预测准确值和学习率将被记录在文件 `.training_precision.log` 里。这一文件用于 [`plot.py`](https://github.com/Blurgyy/spring2019/blob/master/DIP/mnist/exe/plot.py) 绘制图像使用。
  - 训练出的权重被备份在刚才的 `mnist/dmp/` 文件夹下。

### 测试
  - 进入 `exe/` 文件夹进行测试
    ```bash
    python3 validate.py
    ```
    要指定权重文件，使用
    ```bash
    python3 validate.py $WEIGHTS_PATH # 命令行参数是权重文件的路径
    # 如
    # python3 validate.py ../dmp/w.pickle
    ```

### 绘制图像
  - 进入 `exe/` 文件夹进行图像绘制
    ```bash
    python3 plot.py
    ```
    默认使用 `exe/` 文件夹下的文件 `.training_precision.log` 作为数据进行绘制。
    - 要指定绘制图像时使用的数据，使用
      ```bash
      python3 plot.py $LOG_PATH # 命令行参数是记录文件的路径
      # 如
      # python3 plot.py .sigmoid.log
      ```
    - 绘制出的图像中，蓝色的是训练集上的准确率变化曲线，黄色的是学习率(learning rate)在迭代过程中的变化曲线。
      
	  E.g.
    
      ![sigmoid](http://106.14.194.215/imghost/mnist_linear_classifier/sigmoid.png "sigmoid")

> 我该去复习了

