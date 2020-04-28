#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# WuBin modify in 20200419

from __future__ import (division, print_function, absolute_import, unicode_literals)

import argparse
import os
import os.path
import sys
import time
import json

try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3


################################################################################

USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')

zone={
    'SouthChinaSea' :{
        'name':"中国南海",
        'slat':'12',
        'nlat':'13',
        'wlon':'112',
        'elon':'113',
    },
    'WesternPacific':{
        'name': "西太平洋",
        'slat':'17',
        'nlat':'18',
        'wlon':'132',
        'elon':'133',
    },
    'SouthPacific':{
        'name': "南太平洋",
        'slat': '-25',
        'nlat': '-24',
        'wlon': '-116',
        'elon': '-115',
    },
    'Hawaii':{
        'name': "夏威夷海",
        'slat': '19',
        'nlat': '20',
        'wlon': '-157',
        'elon': '-156',
    },
    'Antarctica':{
        'name': "南极冰原",
        'slat': '-68',
        'nlat': '-67',
        'wlon': '86',
        'elon': '87',
    }
}
def chunk_report(bytes_so_far, total_size, speed):
    percent = float(bytes_so_far) / total_size
    percent = round(percent*100, 2)
    print("\r### Downloaded %d of %d bytes [%0.2f%%, %.2fKB/s]" %
          (bytes_so_far, total_size, percent, speed),
          flush=True, end="")


def chunk_read(response, out, chunk_size=8192):
    total_size = response.info()['Content-Length'].strip()
    total_size = int(total_size)
    bytes_so_far = 0
    if isinstance(out, str):
        outf = open(out, 'wb')
    else:
        outf = out

    sec_size = 0
    t0 = time.time()
    t1 = time.time()
    while True:
        chunk = response.read(chunk_size)
        outf.write(chunk)
        outf.flush()
        bytes_so_far += len(chunk)
        sec_size += len(chunk)
        t2 = time.time()
        if t2 - t1 >= 1.0:
            speed = sec_size / (t2 - t1) / 1024
            chunk_report(bytes_so_far, total_size, speed)
            t1 = t2
            sec_size = 0

        if not chunk:
            break

    t2 = time.time()
    if t2 - t0 > 0.0:
        speed = bytes_so_far / (t2 - t0) / 1024
    else:
        speed = 0.0
    chunk_report(bytes_so_far, total_size, speed)
    print('')

    return bytes_so_far


def http_get(url, headers={}):
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                resp = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                return resp
            except urllib2.HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except urllib2.URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                resp = urlopen(Request(url, headers=headers), context=CTX)
                return resp
            except HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

    except AttributeError as e:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        print('exception:', e)
        return None


def geturl(url, token=None, out=None):
    headers = {'user-agent': USERAGENT}
    if token is not None:
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
                    # shutil.copyfileobj(fh, out)
                    chunk_read(fh, out)
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
                    # shutil.copyfileobj(fh, out)
                    chunk_read(fh, out)
            except HTTPError as e:
                print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                print('HTTP GET error message: %s' % e.message, file=sys.stderr)
            except URLError as e:
                print('Failed to make request: %s' % e.reason, file=sys.stderr)
            return None

    except AttributeError as e:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        print('exception:', e)
        print('[!]use CRUL')
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

def download_files(files, dest, tok,sea_zone,slat,nlat,wlon,elon):
    '''synchronize src url with dest directory'''
    print('download_files:', dest)

    base_url = 'https://ladsweb.modaps.eosdis.nasa.gov'
    print('TOTAL file count:', len(files))

    # use os.path since python 2/3 both support it while pathlib is 3.4+
    idx = -1
    for key, f in files.items():
        # currently we use filesize of 0 to indicate directory
        idx += 1
        filesize = int(f['size'])
        print('--->[%d/%d] %s size:%d' % (idx+1, len(files), f['name'], filesize))
        # print(f['start'],type(f['start']))
        if sea_zone == 'manual':
            zone_path = os.path.join(dest, 'manual_%s_%s_%s_%s' %( slat, nlat, wlon, elon))
        else:
            zone_path = os.path.join(dest,f['name'].split('.')[0]+'_'+zone[sea_zone]['name'])
        startdate = f['start'][0:7]
        datedir = "".join(startdate.split("-"))
        # print(datedir)
        datepath = os.path.join(zone_path,datedir)
        path = os.path.join(datepath,f['name'])
        if not os.path.exists(datepath):
            os.makedirs(datepath)
        url = base_url + f['fileURL']
        try:
            if not os.path.exists(path):
                print('downloading: ', path)
                with open(path, 'w+b') as fh:
                    geturl(url, tok, fh)
                print('OK')
            else:
                local_fsize = os.path.getsize(path)
                if local_fsize != filesize:
                    print('[!]file exists. BUT size error. redownload.')
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                    print('OK')
                else:
                    print('Skipping')
        except IOError as e:
            print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
            sys.exit(-1)


def get_jsonobj_via_url(date_from, date_to, slat, nlat, wlon, elon):
    base = 'https://ladsweb.modaps.eosdis.nasa.gov/api/v1/files/product=MOD03,MOD021KM&collection=61&dateRanges={date_from}..{date_to}&areaOfInterest=x{wlon}y{nlat},x{elon}y{slat}&dayCoverage=true&dnboundCoverage=true'
    headers = {
            'X-Requested-With': 'XMLHttpRequest'
            }
    url = base.format(**locals())
    resp = http_get(url, headers)
    if not resp:
        print('ERROR http_get')
        return None
    text = resp.read()
    jsonobj = json.loads(text)
    key_list = list(jsonobj.keys())
    key_list.sort()
    field_list = [
            'name',
            'fileURL',
            'ESDT',
            'collection',
            'start',
            'end',
            'size',
            'GRingLatitude1',
            'GRingLatitude2',
            'GRingLatitude3',
            'GRingLatitude4',
            'GRingLongitude1',
            'GRingLongitude2',
            'GRingLongitude3',
            'GRingLongitude4',
            'status',
            ]
    print('save files json to content.csv')
    with open('content.csv', 'w') as f:
        f.write(','.join(field_list))
        f.write('\n')
        for k in key_list:
            obj = jsonobj[k]
            value_list = [str(obj[name]) for name in field_list]
            f.write(','.join(value_list))
            f.write('\n')
    return jsonobj


if __name__ == '__main__':
    DESC = "useage: python3 modis_data_download_ll.py  -s 2020-04-01 -e 2020-04-02 -z SouthChinaSea"
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=DESC)
    parser.add_argument('-s', '--start', dest='start', metavar='date_from', help='date_from like 2020-04-01', required=True)
    parser.add_argument('-e', '--end', dest='end', metavar='date_to', help='date_to like 2020-04-01', required=True)
    parser.add_argument('-z', '--zone', dest='zone', metavar='Sea zone', help='zone like SouthChinaSea,WesternPacific,SouthPacific,,Hawaii,Antarctica or manual ',
                        required=True)
    parser.add_argument('-sl', '--slat', dest='slat', metavar='slat', help='number like 10.2', required=False)
    parser.add_argument('-nl', '--nlat', dest='nlat', metavar='nlat', help='number like 20.5', required=False)
    parser.add_argument('-wl', '--wlon', dest='wlon', metavar='wlon', help='number like -120.2', required=False)
    parser.add_argument('-el', '--elon', dest='elon', metavar='elon', help='number like -10.4', required=False)
    args = parser.parse_args(sys.argv[1:])


   # if len(sys.argv) < 7:
    #     print('err!')
    #     print('useage: modis_data_download_ll.py $date_from $date_to $south_lat $north_lat $west_lon $east_lon')
    #     print('$date_from/$date_to format: 2020-04-22')
    #     print('$lat/$lon format: 118.12')
    #     exit(-1)
    # date_from = sys.argv[1]
    # date_to = sys.argv[2]
    # slat = sys.argv[3]
    # nlat = sys.argv[4]
    # wlon = sys.argv[5]
    # elon = sys.argv[6]

    date_from = args.start
    date_to = args.end
    sea_zone = args.zone
    if sea_zone == 'manual':
        slat = args.slat
        nlat = args.nlat
        wlon = args.wlon
        elon = args.elon
    else:
        slat = zone[sea_zone]['slat']
        nlat = zone[sea_zone]['nlat']
        wlon = zone[sea_zone]['wlon']
        elon = zone[sea_zone]['elon']
    print('date[from, to]:', date_from, date_to)
    print('lat[south, north]:', slat, nlat)
    print('lon[west, east]:', wlon, elon)

    print('GET JSON files...')
    # jsonobj = get_jsonobj_via_url('2020-04-20', '2020-04-21', 20, 30, 90, 100)
    jsonobj = get_jsonobj_via_url(date_from, date_to, slat, nlat, wlon, elon)

    if not jsonobj:
        print('get json error')
        exit(-1)

    print('GET JSON files OK')

    outPath = "./modis/data"
    # instrument = ['MOD03', 'MOD021KM']
    # destination = os.path.join(outPath, '%s_%s_%s_%s_%s_%s' % (date_from, date_to, slat, nlat, wlon, elon))
    destination = outPath
    if not os.path.exists(destination):
        os.makedirs(destination)
    # 下载token
    #下载的appkey
    token = "这里换成自己的AppKey"
    download_files(jsonobj, destination, token,sea_zone,slat,nlat,wlon,elon)
