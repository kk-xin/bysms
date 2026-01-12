from django.contrib import admin
from django.urls import path
from sales.views import listorders
from sales.views import listcustomers

import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append("..")

urlpatterns = [
    path('admin/', admin.site.urls),
    # 添加如下的路由记录
    path('orders/', listorders),
    path('customers/', listcustomers),
]