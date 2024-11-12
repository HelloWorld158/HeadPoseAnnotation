import os,sys
import cvSDK.DataIO as dio
def GetDefaultDict():
    dct={
        'gpuid':[-1],
        'mode':'deploy',
        "basic_method":"",
        'build':'build_debug',
        'modelcfg':'',
        'deploypyfile':'',
        'checkpoint':'',
        'img':'',
        'test_img':None,
        'calib_dataset_cfg':None,
        'device':'cpu',
        'log_level':'INFO',
        'show':False,
        'dump_info':False,
        'quant_image_dir':None,
        'quant':False,
        'uri':'192.168.1.1:60000'
    }
    return dct
def deploy_args(cfg,curDir):
    ndct={}
    matchdct={
        #'modelcfg':'model_cfg',
        #'deploypyfile':'deploy_cfg',
        #'checkpoint':'checkpoint',
        'img':'img',
        'test_img':'test_img',
        'calib_dataset_cfg':'calib_dataset_cfg',
        'device':'device',
        'log_level':'log_level',
        'show':'show',
        'dump_info':'dump_info',
        'quant_image_dir':'quant_image_dir',
        'quant':'quant',
        'uri':'uri'
    }
    pyfile=cfg['modelcfg']
    if os.path.isdir(cfg['modelcfg']):
        jsdata=dio.getjsondata(os.path.join(cfg['modelcfg'],'trainConfig.json'))
        pyfile=jsdata['configpyfile']
        if not os.path.isabs(pyfile):
            pyfile=os.path.join(cfg['modelcfg'],jsdata['configpyfile'])
    print(f'model_cfg:{pyfile}')
    ndct['model_cfg']=pyfile
    ndct['deploy_cfg']=cfg['deploypyfile']
    if not os.path.isabs(ndct['deploy_cfg']):
        ndct['deploy_cfg']=os.path.join(curDir,cfg['deploypyfile'])
    ndct['checkpoint']=cfg['checkpoint']
    if ndct['checkpoint'] is None or len(ndct['checkpoint'])==0:
        upDir=os.path.dirname(os.path.dirname(pyfile))
        ndct['checkpoint']=os.path.join(upDir,'workDir/best.pth')
        print(f"use default pth:{ndct['checkpoint']}")
    ndct['work_dir']=os.path.join(curDir,'workDir')
    for kk,vv in matchdct.items():
        ndct[vv]=cfg[kk]
    return ndct
def InitDeployConfigFile(config,curDir):
    if os.path.isabs(config.deploypyfile):
        outfile=config.deploypyfile
    else:
        outfile=os.path.join(curDir,config.deploypyfile)
    if os.path.exists(outfile):
        fp=open(outfile,'r')
        txts=fp.readlines()
        fp.close()
        if len(txts)>10:
            return outfile,False
    fp=open(outfile,'w+')
    basfile=config.basic_method
    if not os.path.isabs(basfile):
        basfile=os.path.join(curDir,basfile)
    wline=f'_base_=[\'{basfile}\']'
    fp.write(wline+'\n')
    fp.close()
    return outfile,True
def DumpConfigFile(config,curDir):
    outfile,flag=InitDeployConfigFile(config,curDir)
    from mmengine.config import Config
    depconfig=Config.fromfile(outfile)
    if flag:
        depconfig.dump(outfile)
        print('---------------------------------confirm---------------------------------------------')
        print('Please Change MMDeployConfig.py in FineTune,Steady in deal,Nextime will Truly deal model')
        print('-------------------------------------------------------------------------------------')
        exit()
    return depconfig
def ReplaceArgFunc(mode):
    return mode.replace('.','_')
def GetAbsModelCfg(modelcfg):
    pyfile=modelcfg
    if os.path.isdir(modelcfg):
        jsdata=dio.getjsondata(os.path.join(modelcfg,'trainConfig.json'))
        pyfile=jsdata['configpyfile']
        if not os.path.isabs(pyfile):
            pyfile=os.path.join(modelcfg,jsdata['configpyfile'])
    return pyfile
def GetAbsDeployCfg(deploycfg,curDir):
    pyfile=deploycfg
    if not os.path.isabs(deploycfg):
        pyfile=os.path.join(curDir,deploycfg)
    return pyfile