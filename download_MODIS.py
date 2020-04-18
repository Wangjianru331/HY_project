# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:50:17 2020

@author: xc
"""

# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
from bs4 import BeautifulSoup

class MODIS(object):
    def __init__(self, geolocated, date,username,password):
        self.geolocated = geolocated  # "94.1,40.05,94.3,40.2"   format: minLat,minLon,maxLat,maxLon
        self.date = date              # '2013-05-27'             format:YYYY-MM-DD,2013-05-27
        self.username = username
        self.password = password
#    def setUp(self):
#        self.driver = webdriver.Firefox()
#        self.driver.implicitly_wait(30)
#        self.base_url = "https://www.baidu.com/"
#        self.verificationErrors = []
#        self.accept_next_alert = True
    
    def Get_id(self,t):
        driver = webdriver.Firefox()
        driver.implicitly_wait(30)
        driver.get("https://ladsweb.modaps.eosdis.nasa.gov/search/order/4/MOD021KM--61,MOD03--61/"+self.date+".."+self.date+"/D/"+self.geolocated)  
        #click select all
        time.sleep(t)
        ele1 = driver.find_element_by_id("tab4SelectAllLink")
        #print(ele.is_displayed())
        driver.execute_script("arguments[0].click();", ele1)       
        #click review and order
        time.sleep(t)
        ele2 = driver.find_element_by_id("sub_btn_tab5")
        driver.execute_script("arguments[0].click();", ele2)    
        
        #登录
        driver.find_element_by_id("username").send_keys(self.username)
        driver.find_element_by_id("password").send_keys(self.password)
        driver.find_element_by_css_selector("input[value='Log in']").click()
        
        #submit order
        time.sleep(t)
        ele3 = driver.find_element_by_id("tab5BtnSubmit")
        driver.execute_script("arguments[0].click();", ele3)       
        #click find data
        driver.find_element_by_link_text("Find Data").click()     
        #click 
        time.sleep(t)
        ele4= driver.find_element_by_id("btn_mainPast")
        driver.execute_script("arguments[0].click();", ele4) 
        
        time.sleep(t)
        
        #获取订单
        orderid= []
        dates = []
        rawurl = driver.current_url  # 获取当前URL
        driver.get(rawurl)  # 获取网页
        time.sleep(2*t)  # 等待加载完成
        page = driver.page_source
        soup = BeautifulSoup(page, 'lxml')  # 解析网页
        #抓取下单时间
        tags_dates = soup.find_all('div',class_='poOrderDate')
        for date in tags_dates:
            dates.append(date.contents[0])
        
        #抓取orderid
        tags_ids = soup.find_all('div',class_='poOrderItemStatus')
        for ids in tags_ids:
            orderid.append(ids.contents[0].split()[0])
        return dict(zip(orderid, dates))
       
    def is_element_present(self, how, what):
        try: self.driver.find_element(by=how, value=what)
        except NoSuchElementException as e: return False
        return True
    
    def is_alert_present(self):
        try: self.driver.switch_to_alert()
        except NoAlertPresentException as e: return False
        return True
    
    def close_alert_and_get_its_text(self):
        try:
            alert = self.driver.switch_to_alert()
            alert_text = alert.text
            if self.accept_next_alert:
                alert.accept()
            else:
                alert.dismiss()
            return alert_text
        finally: self.accept_next_alert = True
    
    def tearDown(self):
        self.driver.quit()
        self.assertEqual([], self.verificationErrors)
    
if __name__ == "__main__":
    information = MODIS("117.0,19,119.0,15","2017-04-02",'yourusername','yourpassword')
    t=5
    print(information.Get_id(5))
    