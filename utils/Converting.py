#! -*- coding: utf-8 -*-
import sys
sys.path.append('./utils/function')
import Converting

from abc import ABCMeta, abstractmethod

class Converting(ABCMeta):
    def __init__(verbose = False):
        self.verbose = verbose

    @abstractmethod
    def forward(self, df, cols = None):
        pass

    def __call__(self, df, cols):
        self.forward(df, cols)

####################
## table create  ###
####################
class ConvertToFreqTable(Converting):
    """
    原本记录的是物品名称以及对应的销售日期，通过这个方法可以统计，在每个时间点该物品出现的次数。
    """
    def __init__(self, verbose = False):
        super(ConvertToFreqTable, self).__init__(verbose)

    def forward(df, cols):
        return Converting.to_freq_table(df, cols)

class ConvertToAuftragTable(Converting):
    """
    原本每个item记录的是，转单号，物品名字以及其他信息，通过这个方法可以把同一账单的多个物品进行统合，也就是同一个账单都包含有哪些物品
    """
    def __init__(self, verbose = False):
        super(ConvertToAuftragTable, sefl).__init__(verbose)

    def forward(df, cols):
        return Convrting.to_auftrag_table(df, cols)

class ConvertToCountTable(Converting):
    """
    convert a object column to a count table, this table include two columns, 
    The first column is the target object attribute, and the second column is the count of the value
    We sort the table according to the count of the value
    """
    def __init__(self, verbose = False):
        super(ConvertToCountTable, self).__init__(verbose)

    def forward(df, cols):
        """
        input:
            cols list of string:
                only take the first value as the target value, 
        output:
            out dataframe:
                dataframe contains two columns, one for target attribute one for count. dataframe sorted by count
        """
        return Converting.to_count_table(df, cols)

class ConvertToStatisticTable(Converting):
    """
    like the describe function, but with more statistic value
    """
    def __init__(self, include_object = True, verbose = False):
        super(ConvertToStatisticTable, self).__init__(verbose)
        self._ino = include_object

    def forward(df, cols):
        #TODO implement the method
        return Converting.to_statistic_table(df, cols, self._ino)
