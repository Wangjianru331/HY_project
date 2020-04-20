
import subprocess,os,glob
from dotenv import load_dotenv

if __name__=='__main__':
    ifiles=glob.glob('/mnt/hgfs/share/test/MOD021KM*.hdf')
    with open('/mnt/hgfs/share/test/processing_log.txt','w') as f:
        for i,ifile in enumerate(ifiles):
            if len(os.path.basename(ifile))!=44:continue
            dir=os.path.dirname(ifile)
            name=os.path.basename(ifile)
            ID=name[9:27]
            #print(dir+'/'+'MOD03.'+ID+'*hdf')
            geofile=glob.glob(dir+'/'+'MOD03.'+ID+'*hdf')[0]

            ofile=dir+'/'+name[0:27]+'_seadas1.hdf'
            maskland = 1
            cloudland = 1
            south=32
            west=28
            north=36
            east=35

            l2prod1 ='["Rrs_vvv",\
"aerindex",\
"angstrom",\
"aot_748",\
"aot_859",\
"aot_869",\
"aot_vvv",\
"brdf",\
"chlor_a",\
"humidity",\
"nLw_748",\
"nLw_859",\
"nLw_869",\
"nLw_vvv",\
"ozone",\
"pressure",\
"sena",\
"senz",\
"sola",\
"solz",\
"water_vapor"]'
            l2prod1

            l2gencmd1='l2gen'+' ifile='+ ifile +' geofile='+ geofile +' ofile='+ ofile +' south='+ \
                     str(south)+' west='+str(west)+' north='+str(north)+' east='+str(east)+\
                     ' l2prod1='+str(l2prod1)
            l2gencmd2 = 'l2gen' + ' ifile=' + ifile + ' geofile=' + geofile + ' ofile=' + ofile

            #l2cmdtest = 'l2gen -h'
            try:

                commands = 'source $OCSSWROOT/OCSSW_bash.env;'+l2gencmd1
                process = subprocess.Popen(commands,executable='/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE,shell=True)
                out, err = process.communicate()
                print("any error?:",err)

            except:
                f.write(ifile+'\n')


