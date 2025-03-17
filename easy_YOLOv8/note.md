
# YOLOv8-Python笔记

### [python的导包操作](https://blog.csdn.net/m0_67575344/article/details/124106709)
### [Python any()和all()的用法](https://blog.csdn.net/sinat_36197913/article/details/103356742)
### [python静态方法、实例方法、类方法使用](https://blog.csdn.net/tianheihei__/article/details/124598675)
### [Python【方法和返回值（Union）联合类型】注解](https://blog.csdn.net/weixin_59131972/article/details/128504652)
### startsWith()方法
- 用来判断当前字符串是否是以另外一个给定的子字符串“开头”的，根据判断结果返回 true 或 false。
### pathlib中Path模块总结
- <https://blog.csdn.net/weixin_42484997/article/details/108594478>
- <https://img-blog.csdnimg.cn/13b8e67030ab42c99649478f888fb67e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASVTnlYzmkKzov5DllrU=,size_16,color_FFFFFF,t_70,g_se,x_16>
### [python @property的介绍与使用](https://zhuanlan.zhihu.com/p/64487092)
### [hasattr() 函数用于判断对象是否包含对应的属性。](https://www.runoob.com/python/python-func-hasattr.html)
### [Python中typing模块详解](https://www.jianshu.com/p/9b6b9a06cd3e)
### [Python \_\_call\_\_()方法（详解版）](http://c.biancheng.net/view/2380.html)
### [Python魔法方法之\_\_getattr\_\_()和getattribute](https://www.py.cn/jishu/jichu/10799.html)
### 

在定义列表是要使用list()因为其为深拷贝，是一个独立的

### 以下这两种推导式的执行顺序不一样
```python
i = 1
l = [i for i in range(10) if i ==1]
for i in range(10) if i ==1 else range(20):
    print(i)
```
