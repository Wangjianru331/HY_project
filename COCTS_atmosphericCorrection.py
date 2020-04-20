# import os
# cmdstr = 'hy1c_l2gen'
# # inputstr = 'ifile=/home/quzhou/HY1C_L2A_batch_run/testData/H1B_OPER_OCT_L1B_20070513T022200_20070513T022200_00457_10.h5'
# inputstr = 'ifile=/mnt/hgfs/share/1b/H1B_OPER_OCT_L1A_20070425T021200_20070425T021200_00200_10.h5'
# # outputstr = 'ofile=/home/quzhou/HY1C_L2A_batch_run/testData/H1B_OPER_OCT_L1B_20070513T022200_20070513T022200_00457_10_test2.h5'
# # outputstr = 'ofile=/home/quzhou/HY1C_L2A_batch_run/testData/H1C_OPER_OCT_L1B_20181231T161500_20181231T162000_01663_10test.h5'
# outputstr = 'ofile='+os.path.dirname(inputstr)[6:]+os.sep+os.path.basename(inputstr)[:-3]+'_l2gen.h5'
# print(outputstr)
# prodstr = 'l2prod="rhot_412 rhot_443 rhot_490 rhot_520 rhot_565 rhot_670 rhot_750 rhot_865 Rrs412 Rrs443 Rrs490 Rrs520 Rrs565 Rrs670 Rrs750 Rrs865 chlor_a Rrc_412 Rrc_443 Rrc_490 Rrc_520 Rrc_565 Rrc_670 Rrc_750 Rrc_865"'
# CMD_STR = cmdstr + ' ' + inputstr + ' ' + outputstr + ' ' + prodstr
# #os.system('export HY1CDATAROOT="/home/song/HY1C_L2A_batch_run/hy1cdata"')
# os.environ['HY1CDATAROOT']='/home/quzhou/HY1C_L2A_batch_run/hy1cdata'
# os.system(CMD_STR)
# os.system('exit')


# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:51:07 2019

@author: song
"""
import glob
import os

path = '/mnt/hgfs/share/1b'
files = glob.glob(path+os.sep+'H1A*.h5')
for i, infile in enumerate(files):
	cmdstr = '/home/quzhou/HY1C_L2A_batch_run/hy1c_l2gen'
	inputstr = 'ifile='+infile #/mnt/hgfs/握手/H1A_OPER_OCT_L1B_20021017T001333_20021017T042936_2199_10.h5'
	outputstr = 'ofile='+os.path.dirname(infile) +os.sep+os.path.basename(infile)[:-3]+'_l2gen.h5'
	prodstr = 'l2prod="rhot_412 rhot_443 rhot_490 rhot_520 rhot_565 rhot_670 rhot_750 rhot_865 Rrs412 Rrs443 Rrs490 Rrs520 Rrs565 Rrs670 Rrs750 Rrs865 chlor_a Rrc_412 Rrc_443 Rrc_490 Rrc_520 Rrc_565 Rrc_670 Rrc_750 Rrc_865"'
	CMD_STR = cmdstr + ' ' + inputstr + ' ' + outputstr + ' ' + prodstr
	#os.system('export HY1CDATAROOT="/home/song/HY1C_L2A_batch_run/hy1cdata"')
	#os.environ['HY1CDATAROOT']='/home/song/HY1C_L2A_batch_run/hy1cdata'
	os.environ['HY1CDATAROOT']='/home/quzhou/HY1C_L2A_batch_run/hy1cdata'
	os.system(CMD_STR)

	os.system('exit')