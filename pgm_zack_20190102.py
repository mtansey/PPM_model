# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:08:40 2019

@author: zleady
"""
# standard libraries
import os
import sys
import datetime
import logging
import argparse
import math

# data manipilation libraries
# import numpy as np
import pandas as pd

# threading libraries
from multiprocessing import Pool as ThreadPool
from functools import partial

# graphing libraries
from plotly import tools
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio


def CreateLogger(log_file):
    """ Zack's Generic Logger function to create onscreen and file logger

    Parameters
    ----------
    log_file: string
        `log_file` is the string of the absolute filepathname for writing the
        log file too which is a mirror of the onscreen display.

    Returns
    -------
    logger: logging object

    Notes
    -----
    This function is completely generic and can be used in any python code.
    The handler.setLevel can be adjusted from logging.INFO to any of the other
    options such as DEBUG, ERROR, WARNING in order to restrict what is logged.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
# create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# create error file handler and set level to info
    handler = logging.FileHandler(log_file,  "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class ManagementInputs:

    def __init__(self, management_file):
        self.created_from = management_file
        management_df = self.read_management_inputs(self.created_from)
        self.sowing_year = management_df.loc[0, 'sowing_year']
        self.sowing_doy = management_df.loc[0, 'sowing_doy']
        self.sowing_density = management_df.loc[0, 'sowing_density']

    def read_management_inputs(self, management_inputs_file):
        df = pd.read_csv(management_inputs_file, header=0, skiprows=[1])
        return df

    def get_sowing_density(self):
        sowing_density = float(self.sowing_density)
        return sowing_density

    def get_sowing_year(self):
        sowing_year = int(self.sowing_year)
        return sowing_year

    def get_sowing_doy(self):
        sowing_doy = int(self.sowing_doy)
        return sowing_doy


def read_weather_inputs(weather_inputs_file):
    df = pd.read_csv(weather_inputs_file, header=0, skiprows=[1])
    return df


def read_crop_inputs(crop_inputs_file):
    df = pd.read_csv(crop_inputs_file, header=0, index_col=0, skiprows=[1])
    return df


class Crop:
    def __init__(self, crop_name, crop_df, mi_obj, write_path):
        self.crop_name = crop_name
        self.sowing_year = mi_obj.get_sowing_year()
        self.sowing_doy = mi_obj.get_sowing_doy()
        self.write_path = write_path
        """Initialize values from crop_df"""
        # From Phenology
        self.TBD = crop_df.loc[crop_name, 'TBD']
        self.TP1D = crop_df.loc[crop_name, 'TP1D']
        self.TP2D = crop_df.loc[crop_name, 'TP2D']
        self.TCD = crop_df.loc[crop_name, 'TCD']
        self.tuSOWEMR = crop_df.loc[crop_name, 'tuSOWEMR']
        self.tuEMRTLM = crop_df.loc[crop_name, 'tuEMRTLM']
        self.tuTLMBSG = crop_df.loc[crop_name, 'tuTLMBSG']
        self.tuBSGTSG = crop_df.loc[crop_name, 'tuBSGTSG']
        self.tuTSGMAT = crop_df.loc[crop_name, 'tuTSGMAT']
        # From CropLAI
        self.PHYL = crop_df.loc[crop_name, 'phyl']
        self.PLACON = crop_df.loc[crop_name, 'PLACON']
        self.PLAPOW = crop_df.loc[crop_name, 'PLAPOW']
        self.SLA = crop_df.loc[crop_name, 'SLA']
        # From DMProduction
        self.TBRUE = crop_df.loc[crop_name, 'TBRUE']
        self.TP1RUE = crop_df.loc[crop_name, 'TP1RUE']
        self.TP2RUE = crop_df.loc[crop_name, 'TP2RUE']
        self.TCRUE = crop_df.loc[crop_name, 'TCRUE']
        self.KPAR = crop_df.loc[crop_name, 'KPAR']
        self.IRUE = crop_df.loc[crop_name, 'IRUE']
        # From DMDistribution
        self.FLF1A = crop_df.loc[crop_name, 'FLF1A']
        self.FLF1B = crop_df.loc[crop_name, 'FLF1B']
        self.WTOPL = crop_df.loc[crop_name, 'WTOPL']
        self.FLF2 = crop_df.loc[crop_name, 'FLF2']
        self.FRTRL = crop_df.loc[crop_name, 'FRTRL']
        self.GCC = crop_df.loc[crop_name, 'GCC']
        """ Initialize other values """
        # From Phenology
        self.tuEMR = self.tuSOWEMR
        self.tuTLM = self.tuEMR + self.tuEMRTLM
        self.tuBSG = self.tuTLM + self.tuTLMBSG
        self.tuTSG = self.tuBSG + self.tuBSGTSG
        self.tuMAT = self.tuTSG + self.tuTSGMAT
        self.DAP = 0
        self.CTU = 0
        # From CropLAI
        self.MSNN = 1
        self.PLA1 = 0
        self.PLA2 = 0
        self.LAI = 0
        self.MXLAI = 0
        # From DMDistribution
        self.WLF = 0.5
        self.WST = 0.5
        self.WVEG = self.WLF + self.WST
        self.WGRN = 0
        """ Zack Initialized """
        # phenology_process
        self.DTU = 0
        self.DTEMR = 0
        self.DTTLM = 0
        self.DTBSG = 0
        self.DTTSG = 0
        self.MAT = 0
        # croplai_process
        self.GLAI = 0
        self.DLAI = 0
        self.GLF = 0
        self.PDEN = mi_obj.get_sowing_density()
        self.BSGLAI = 0
        # dmproduction_process
        self.FINT = 0
        self.DDMP = 0
        self.TCFRUE = 0
        # dmdistribution_process
        self.WTOP = 0
        self.SGR = 0
        self.GST = 0
        self.BSGDM = 0

    def get_crop_name(self):
        return self.crop_name

    def get_mat(self):
        return int(self.MAT)

    def get_write_path(self):
        return self.write_path
    
    def get_sowing_year(self):
        return self.sowing_year
    
    def get_sowing_doy(self):
        return self.sowing_doy

    def phenology_process(self, TMP):
        # Temperature Process
        if TMP <= self.TBD or TMP >= self.TCD:
            tempfun = 0
        elif TMP > self.TBD and TMP < self.TP1D:
            tempfun = (TMP - self.TBD) / (self.TP1D - self.TBD)
        elif TMP > self.TP2D and TMP < self.TCD:
            tempfun = (self.TCD - TMP) / (self.TCD - self.TP2D)
        elif TMP >= self.TP1D and TMP <= self.TP2D:
            tempfun = 1
        else:
            logging.error('Crop: {} failed temperature process'
                          .format(self.crop_name))
        self.DTU = (self.TP1D - self.TBD) * tempfun
        self.CTU += self.DTU
        self.DAP += 1
        # CTU Process
        if self.CTU < self.tuEMR:
            self.DTEMR = self.DAP + 1
        if self.CTU < self.tuTLM:
            self.DTTLM = self.DAP + 1
        if self.CTU < self.tuBSG:
            self.DTBSG = self.DAP + 1
        if self.CTU < self.tuTSG:
            self.DTTSG = self.DAP + 1
        if self.CTU < self.tuMAT:
            self.DTMAT = self.DAP + 1
        if self.CTU > self.tuMAT:
            self.MAT = 1

    def croplai_process(self):
        # Yesterday LAI to intercept PAR today
        self.LAI = self.LAI + self.GLAI - self.DLAI
        if self.LAI < 0:
            self.LAI = 0
        if self.LAI > self.MXLAI:
            self.MXLAI = self.LAI
        # Daily Increase/Decrease in LAI today
        if self.CTU <= self.tuEMR:
            self.GLAI = 0
            self.DLAI = 0
        elif self.CTU > self.tuEMR and self.CTU <= self.tuTLM:
            INODE = self.DTU / self.PHYL
            self.MSNN += INODE
            self.PLA2 = self.PLACON * self.MSNN**self.PLAPOW
            self.GLAI = ((self.PLA2 - self.PLA1) * self.PDEN / 10000)
            self.PLA1 = self.PLA2
            self.DLAI = 0
        elif self.CTU > self.tuTLM and self.CTU <= self.tuBSG:
            self.GLAI = self.GLF * self.SLA
            self.BSGLAI = self.LAI
            self.DLAI = 0
        elif self.CTU > self.tuBSG:
            self.GLAI = 0
            self.DLAI = self.DTU / (self.tuMAT - self.tuBSG) * self.BSGLAI

    def dmproduction_process(self, TMP, SRAD):
        # Adjustment of RUE
        if TMP <= self.TBRUE or TMP >= self.TCRUE:
            self.TCFRUE = 0
        elif TMP > self.TBRUE or TMP < self.TP1RUE:
            self.TCFRUE = (TMP - self.TBRUE) / (self.TP1RUE - self.TBRUE)
        elif TMP > self.TP2RUE and TMP < self.TCRUE:
            self.TCFRUE = (self.TCRUE - TMP) / (self.TCRUE - self.TP2RUE)
        elif TMP >= self.TP1RUE and TMP <= self.TP2RUE:
            self.TCFRUE = 1
        else:
            logging.error('Dry matter process failed on {}'
                          .format(self.crop_name))
        RUE = self.IRUE * self.TCFRUE
        # Daily dry matter production
        self.FINT = 1 - math.exp(-self.KPAR*self.LAI)
        self.DDMP = SRAD * 0.48 * self.FINT * RUE

    def dmdistribution_process(self):
        if self.CTU <= self.tuEMR or self.CTU > self.tuTSG:
            self.DDMP = 0
            self.GLF = 0
            self.GST = 0
            TRANSL = 0
            self.SGR = 0
        elif self.CTU > self.tuEMR and self.CTU <= self.tuTLM:
            if self.WTOP < self.WTOPL:
                FLF1 = self.FLF1A
            else:
                FLF1 = self.FLF1B
            self.GLF = FLF1 * self.DDMP
            self.GST = self.DDMP - self.GLF
            self.SGR = 0
        elif self.CTU > self.tuTLM and self.CTU <= self.tuBSG:
            self.GLF = self.FLF2 * self.DDMP
            self.GST = self.DDMP - self.GLF
            self.SGR = 0
            self.BSGDM = self.WTOP
        elif self.CTU > self.tuBSG and self.CTU <= self.tuTSG:
            self.GLF = 0
            self.GST = 0
            TRLDM = self.BSGDM * self.FRTRL
            TRANSL = self.DTU / (self.tuTSG - self.tuBSG) * TRLDM
            self.SGR = (self.DDMP + TRANSL) * self.GCC
        self.WLF += self.GLF
        self.WST += self.GST
        self.WGRN += self.SGR
        self.WVEG = self.WVEG + self.DDMP - (self.SGR / self.GCC)
        self.WTOP = self.WVEG + self.WGRN

    def ini_df_outputs(self):
        self.df_daily_outputs = pd.DataFrame(columns=['Yr', 'DOY', 'DAP',
                                                      'TMP',
                                                      'DTU', 'CTU', 'MSNN',
                                                      'GLAI', 'DLAI', 'LAI',
                                                      'TCFRUE', 'FINT', 'DDMP',
                                                      'GLF', 'GST', 'SGR',
                                                      'WLF', 'WST', 'WVEG',
                                                      'WGRN', 'WTOP',
                                                      'Harvest_index'])
        # harvest index = WGRN / WTOP * 100
        self.df_summary_outputs = pd.DataFrame(columns=['DTEMR', 'DTTLM',
                                                        'DTBSG', 'DTTSG',
                                                        'DTMAT', 'MXLAI',
                                                        'BSGLAI', 'BSGDM',
                                                        'WTOP', 'WGRN',
                                                        'Harvest_index'])

    def update_daily_outputs(self, row, TMP, Yr, DOY):
        harvest_index = (self.WGRN / self.WTOP) * 100
        self.df_daily_outputs.loc[row, :] = [Yr, DOY, self.DAP, TMP, self.DTU,
                                             self.CTU, self.MSNN, self.GLAI,
                                             self.DLAI, self.LAI, self.TCFRUE,
                                             self.FINT, self.DDMP, self.GLF,
                                             self.GST, self.SGR, self.WLF,
                                             self.WST, self.WVEG, self.WGRN,
                                             self.WTOP, harvest_index]

    def update_summary_outputs(self):
        # harvest index = WGRN / WTOP * 100
        harvest_index = (self.WGRN / self.WTOP) * 100
        self.df_summary_outputs.loc[0, :] = [self.DTEMR, self.DTTLM,
                                             self.DTBSG, self.DTTSG,
                                             self.DTMAT, self.MXLAI,
                                             self.BSGLAI, self.BSGDM,
                                             self.WTOP, self.WGRN,
                                             harvest_index]

    def gen_graphs(self):
        self.graphs_output = []
        self.graphs_ids = []
        df = self.df_daily_outputs
        x_graph = df['DAP'].values.tolist()
        graph_y_vars = ['DTU', 'CTU', 'LAI', ['DDMP', 'SGR'],
                        ['WVEG', 'WGRN', 'WTOP']]
        for y_var in graph_y_vars:
                xaxis_title = 'DAP'
                if isinstance(y_var, list):
                    # print('found: {} as list'.format(y_var))
                    yname = '_'.join(y_var)
                    title_name = ' and '.join(y_var)
                    title = '{} vs. DAP'.format(title_name)
                    yaxis_title = '{}'.format(title_name)
                    data = []
                    for y in y_var:
                        y_graph = df[y].values.tolist()
                        trace_temp = go.Scatter(x=x_graph, y=y_graph,
                                                mode='lines+markers',
                                                name='{}'.format(y))
                        data.append(trace_temp)
                else:
                    # print('not list {}'.format(y_var))
                    yname = str(y_var)
                    title = '{} vs. DAP'.format(yname)
                    yaxis_title = '{}'.format(yname)
                    data = []
                    y_graph = df[y_var].values.tolist()
                    trace_temp = go.Scatter(x=x_graph, y=y_graph,
                                            mode='lines+markers',
                                            name='{}'.format(y_var))
                    data.append(trace_temp)
                layout = go.Layout(title=title, xaxis=dict(title=xaxis_title),
                                   yaxis=dict(title=yaxis_title),
                                   autosize=True)
                fig = go.Figure(data=data, layout=layout)
                self.graphs_output.append(fig)
                self.graphs_ids.append('{}_DAP'.format(yname))

    def write_graph_image(self):
        basic_path = self.write_path
        if not os.path.exists(os.path.join(basic_path, 'graph_images')):
            os.mkdir(os.path.join(basic_path, 'graph_images'))
        for i, f in zip(self.graphs_ids, self.graphs_output):
            output_path = os.path.join(basic_path, 'graph_images',
                                       '{}.png'.format(i))
            pio.write_image(f, output_path)

    def write_graph_html(self):
        basic_path = self.write_path
        if not os.path.exists(os.path.join(basic_path, 'graph_htmls')):
            os.mkdir(os.path.join(basic_path, 'graph_htmls'))
        for i, f in zip(self.graphs_ids, self.graphs_output):
            output_path = os.path.join(basic_path, 'graph_htmls',
                                       '{}.html'.format(i))
            plot(f, filename=output_path, auto_open=False, show_link=False,
                 config=dict(displaylogo=False))

    def write_outputs(self):
        basic_path = self.write_path
        daily_outputs_path = os.path.join(basic_path, '{}_daily.csv'
                                          .format(self.crop_name))
        self.df_daily_outputs.to_csv(daily_outputs_path, sep=",")
        summary_outputs_path = os.path.join(basic_path, '{}_summary.csv'
                                            .format(self.crop_name))
        self.df_summary_outputs.to_csv(summary_outputs_path, sep=",")


def add_weather_tmp(wi_df):
    wi_df['TMP'] = (wi_df['TMAX'] + wi_df['TMIN']) / 2
    wi_df = wi_df.round({'TMP': 2})
    return wi_df


def ProcessMain(N_crop, weather_df):
    """GoSub ManagInputs --> mi_obj = pyear, pdoy, PDEN
    GoSub InitialsHeaders --> ini values == 0, Output DF setup
    GoSub FindSowingDate --> checks to make sure weather data has sowing date
    Do Until MAT = 1
        GoSub Weather --> wi_df & creates TMP = (TMAX + TMIN) / 2
        GoSub Phenology
        GoSub CropLAI
        GoSub DMProduction
        GoSub DMDistribution
        GoSub DailyPrintOut --> Output DF write
    Loop
    GoSub SummaryPrintOut --> aggregate DF Summary"""
    id1 = os.getppid()
    id2 = os.getpid()
    crop_name = N_crop.get_crop_name()
    N_crop.ini_df_outputs()
    mat_trigger = N_crop.get_mat()
    sowing_year = N_crop.get_sowing_year()
    sowing_doy = N_crop.get_sowing_doy()
    i = weather_df.index[(weather_df['YEAR'] == sowing_year) &
                         (weather_df['DOY'] == sowing_doy)].tolist()[0]
    while mat_trigger == 0:
        TMP = weather_df.loc[i, 'TMP']
        # print(TMP)
        SRAD = weather_df.loc[i, 'SRAD']
        Yr = weather_df.loc[i, 'YEAR']
        DOY = weather_df.loc[i, 'DOY']
        N_crop.phenology_process(TMP)
        mat_trigger += N_crop.get_mat()
        N_crop.croplai_process()
        N_crop.dmproduction_process(TMP, SRAD)
        N_crop.dmdistribution_process()
        N_crop.update_daily_outputs(i, TMP, Yr, DOY)
        i += 1
    N_crop.update_summary_outputs()
    N_crop.gen_graphs()
    N_crop.write_graph_image()
    N_crop.write_graph_html()
    N_crop.write_outputs()
    return [id1, id2, crop_name, 'complete']


if __name__ == "__main__":
    # begin runtime clock
    start = datetime.datetime.now()
    # determine the absolute file pathname of this *.py file
    abspath = os.path.abspath(__file__)
    # from the absolute file pathname determined above,
    # extract the directory path
    dir_name = os.path.dirname(abspath)
    # initiate logger
    log_file = os.path.join(dir_name, 'pgm_zack_{}.log'
                            .format(str(start.date())))
    CreateLogger(log_file)
    # create the command line parser object from argparse
    parser = argparse.ArgumentParser()
    # set the command line arguments available to user's
    parser.add_argument("--management_inputs", "-mi", type=str,
                        help="Provide the full pathname of runtime inputs \
                        file for the model run")
    parser.add_argument("--weather_inputs", "-wi", type=str,
                        help="Provide the full pathname of the weather inputs \
                        file for the model run")
    parser.add_argument("--crop_inputs", "-ci", type=str,
                        help="Provide the full pathname of the crop inputs \
                        file for the model run")
    parser.add_argument("--write", "-w", type=str,
                        help="Provide the full folder pathname for the \
                        output data files")
    parser.add_argument("--num_threads", "-mp", type=int,
                        help="Provide the number of threads to use for \
                        multiprocessing of crops")
    # create an object of the command line inputs
    args = parser.parse_args()
    # read the command line inputs into a Python dictionary
    # e.g. ini_dict.get("write") : 'C:\Users\zleady\Desktop\output.csv'
    ini_dict = vars(args)
    for k in ini_dict.keys():
        logging.info('Initialization Dict key: {} \n value: {}'
                     .format(k, ini_dict.get(k)))
    mi_obj = ManagementInputs(ini_dict.get("management_inputs"))
    logging.info('Managment Inputs: \n year:{} \n doy:{} \n density: {}'
                 .format(mi_obj.sowing_year, mi_obj.sowing_doy,
                         mi_obj.sowing_density))
    wi_df = read_weather_inputs(ini_dict.get("weather_inputs"))
    wi_df = add_weather_tmp(wi_df)
    logging.info('{}'.format(wi_df))
    ci_df = read_crop_inputs(ini_dict.get("crop_inputs"))
    logging.info('{}'.format(ci_df))
    logging.info('Found {} weather days'.format(len(wi_df)))
    logging.info('Found {} number of crops'.format(len(ci_df)))
    logging.info('Found {} number of threads requested by user'
                 .format(ini_dict.get("num_threads")))
    crop_name_lst = ci_df.index.values.tolist()
    if ini_dict.get("num_threads") > 1:
        thread_pool = ThreadPool(ini_dict.get("num_threads"))
        crop_obj_lst = []
        for c in crop_name_lst:
            print(c)
            temp_C = Crop(c, ci_df, mi_obj,
                          ini_dict.get("write"))
            crop_obj_lst.append(temp_C)
            print(thread_pool.map(partial(ProcessMain,
                                          weather_df=wi_df), crop_obj_lst))
    else:
        for c in crop_name_lst:
            print(c)
            temp_C = Crop(c, ci_df, mi_obj,
                          ini_dict.get("write"))
            print(ProcessMain(temp_C, wi_df))

    elapsed_time = datetime.datetime.now() - start
    logging.info('Runtime: {}'.format(str(elapsed_time)))
