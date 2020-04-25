
# -*- coding: utf-8 -*-
# script supports either python2 or python3
#
# Attempts to do HTTP Gets with urllib2(py2) urllib.requets(py3) or subprocess
# if tlsv1.1+ isn't supported by the python ssl module
#
# Will download csv or json depending on which python module is available
#

from __future__ import (division, print_function, absolute_import, unicode_literals)

import argparse
import os
import os.path
import shutil
import sys

try:
    from StringIO import StringIO  # python2
except ImportError:
    from io import StringIO  # python3


from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
from bs4 import BeautifulSoup
import time


class GetOrder(object):
    def __init__(self, location, date, username, password, sleeptime):
        self.location = location  # "94.1,40.05,94.3,40.2"   format: minLat,minLon,maxLat,maxLon
        self.date = date  # '2013-05-27'             format:YYYY-MM-DD,2013-05-27
        self.username = username
        self.password = password
        self.sleeptime = sleeptime

    def get_id(self):
        driver = webdriver.Firefox()
        driver.implicitly_wait(30)
        driver.get(
            "https://ladsweb.modaps.eosdis.nasa.gov/search/order/4/MOD021KM--61,MOD03--61/" + self.date + "/D/" + self.location)
        # click select all
        time.sleep(self.sleeptime)
        ele1 = driver.find_element_by_id("tab4SelectAllLink")
        # print(ele.is_displayed())
        driver.execute_script("arguments[0].click();", ele1)
        # click review and order
        time.sleep(self.sleeptime)
        ele2 = driver.find_element_by_id("sub_btn_tab5")
        driver.execute_script("arguments[0].click();", ele2)

        # 登录
        driver.find_element_by_id("username").send_keys(self.username)
        driver.find_element_by_id("password").send_keys(self.password)
        driver.find_element_by_css_selector("input[value='Log in']").click()

        # submit order
        time.sleep(self.sleeptime)
        ele3 = driver.find_element_by_id("tab5BtnSubmit")
        driver.execute_script("arguments[0].click();", ele3)
        # click find data
        driver.find_element_by_link_text("Find Data").click()
        # click
        time.sleep(self.sleeptime)
        ele4 = driver.find_element_by_id("btn_mainPast")
        driver.execute_script("arguments[0].click();", ele4)

        time.sleep(self.sleeptime)

        # 获取订单
        orderid = []
        dates = []
        rawurl = driver.current_url  # 获取当前URL
        driver.get(rawurl)  # 获取网页
        time.sleep(2 * self.sleeptime)  # 等待加载完成
        page = driver.page_source
        soup = BeautifulSoup(page, 'lxml')  # 解析网页
        # 抓取下单时间
        tags_dates = soup.find_all('div', class_='poOrderDate')
        for date in tags_dates:
            dates.append(date.contents[0])

        # 抓取orderid
        tags_ids = soup.find_all('div', class_='poOrderItemStatus')
        for ids in tags_ids:
            orderid.append(ids.contents[0].split()[0])
        return dict(zip(orderid, dates))

    def is_element_present(self, how, what):
        try:
            self.driver.find_element(by=how, value=what)
        except NoSuchElementException as e:
            return False
        return True

    def is_alert_present(self):
        try:
            self.driver.switch_to_alert()
        except NoAlertPresentException as e:
            return False
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
        finally:
            self.accept_next_alert = True

    def tearDown(self):
        self.driver.quit()
        self.assertEqual([], self.verificationErrors)


def geturl(url, token=None, out=None):
    USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')
    headers = {'user-agent': USERAGENT}
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except urllib2.URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                fh = urlopen(Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read().decode('utf-8')
                else:
                    shutil.copyfileobj(fh, out)
            except HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

    except AttributeError:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        import subprocess
        try:
            args = ['curl', '--fail', '-sS', '-L', '--get', url]
            for (k, v) in headers.items():
                args.extend(['-H', ': '.join([k, v])])
            if out is None:
                # python3's subprocess.check_output returns stdout as a byte string
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
        return None


################################################################################





def sync(username, password, tok, location, date, sleeptime, dest):

    '''search source ID'''

    submit = GetOrder(location, date, username, password, sleeptime)
    src = submit.get_id()



    '''synchronize src url with dest directory'''
    try:
        import csv
        files = [f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True)]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))

    # use os.path since python 2/3 both support it while pathlib is 3.4+
    for f in files:
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(dest, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path):
                    print('downloading: ', path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0


def _main(argv):
    DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument('-lo', '--location', dest='location', metavar='LOC',
                        help='the area defined by lat-lon from up-left to low-right, for example:94.1,40.05,94.3,40.2',
                        required=True)
    parser.add_argument('-da', '--date', dest='date', metavar='DAT',
                        help='the date,for example:2013-05-27..2013-05-27',
                        required=True)
    parser.add_argument('-u', '--username', dest='username', metavar='USER',
                        help='username registered on earthdata web',
                        required=True)
    parser.add_argument('-p', '--password', dest='password', metavar='PASS',
                        help='password registered on earthdata web',
                        required=True)
    parser.add_argument('-sl', '--sleeptime', dest='sleeptime', metavar='sleeptime',
                        help='load time of webpage, '
                        'set it according your internet speeds ',
                        required=True)
    # parser.add_argument('-s', '--source', dest='source', metavar='URL', help='Recursively download files at URL',
    #                     required=True)
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIR',
                        help='Store directory structure in DIR', required=True)
    parser.add_argument('-t', '--token', dest='token', metavar='TOK',
                        help='Use app token TOK to authenticate',
                        required=True)
    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    return sync(args.username, args.password, args.token, args.location, args.date,
                args.sleeptime, args.destination)


if __name__ == '__main__':
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)