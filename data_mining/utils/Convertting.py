def show_tools():
    print("""
    In this phase we need to engineering each feature, and there are no shortcut here, but we also provide some methods like to show_skew to help to analyse and fix the feature
    # show figures
    show_kde(df, tg, fs)
    show_skew(df)
    # change data
    fix_skew(df, fs)
    # distribution
    classify_with_clauster(df, tg, fs)

    ps1. 这部分过程比较考验经验，给一个统计图表，能不能从中提取出有用的信息，一定程度上决定了最终结果的好坏。
为了不那么晕，一般都是以多层for循环的方式，逐步进行分析的。最好能够了解不同的plot能够比较好的描述哪方面的信息。比如boxplot比较适用于比较概括的分布的对比，而violineplot则能够比较好的描述
    1. 个人感觉，还是kde图比较能够说明事情，既可以表示分布状况又可以看出目标比例。还有一个就是FacetGrit也比较好，可以分row，col，hue三个维度进行比较。
    2. 另外，以上都是通过理智分析，来获得新的属性，是否存在一个方法，可以让他子集生成有用的新的属性呢、
    3. 现在看到的这些分段，大部分都不是在第一次画图后就确定下来的方案(比如分几段，在哪里分段等等)，一般都有多个备用方案，具体使用哪个，都是由最后预测的正确率来决定的
    4. 判断一个属性是否有存在的必要的标准就是，这个属性是否会对，目标造成影响，最直观的表现就是各个取值的目标率相差是否比较大。

    """)

def show_kde(df,tg,fs):
    """
    apply kdeplot in the selected feature.
    input:
        df: dataframe
        tg: terget feature
        fs: selected features
    process:
        check features' type
        apply kdeplot
    """
    #TODO 
    pass

def show_skew(df):
    """
    show the skewness of each feature in the df.
    When a feature distribution is very skewed ot left, This can lead to overweighting the model with very high values.
    In this case, we have better to transform it with the log function to reduce the skewness and redistribute the data.
    input:
        df: the dataframe
    """
    #TODO
    pass

def fix_skew(df, fs):
    """
    fix the very skewed feature in the df, with log function
    input:
        df: the dataframe
        fs: the feature we need to fix
    """
    #TODO
    pass

def classifiy_with_clauster(df, tg, fs):
    """
    Convert的一个重点就是进行分段，我们可以用clauster的方法来完成这个操作
    input:
        df: the dataframe
        tg: target feature
        fs: the features input
    output:
        nf: the new feature
    """
    #TODO 
    pass
