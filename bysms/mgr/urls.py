from django.urls import path
from mgr import customer

import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append("..")

urlpatterns = [
    path('customers', customer.dispatcher),
]