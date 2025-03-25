from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from astropy import wcs 
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

import numpy as np
import os
import pickle
import configparser
import gc
import copy

import src.detector as tod
import src.loaddata as ld
import src.mapmaker as mp
import src.beam as bm
import src.pointing as pt 

from IPython import embed

class AppWindow(QMainWindow):

    '''
    Class to create the app
    '''

    def __init__(self):
        super().__init__()
        self.title = 'NaMap'

        self.setWindowTitle(self.title)

        self.TabLayout = MainWindowTab(self)
        self.setCentralWidget(self.TabLayout)

        self.current_name()
        self.TabLayout.tab1.experiment.activated[str].connect(self.current_name)

        menubar = self.menuBar()

        self.beammenu = menubar.addMenu('Beam Functions')
        beamfit = QAction('Fitting Parameters', self)
        self.beammenu.addAction(beamfit)
        beamfit.triggered.connect(self.beam_fit_param_menu)
        
        self.detmenu = menubar.addMenu('TOD Functions')
        todoff = QAction('Time offset between TOD and Pointing',  self)
        self.detmenu.addAction(todoff)
        todoff.triggered.connect(self.tod_func_offset)
        todproc = QAction('TOD processing parameters',  self)
        self.detmenu.addAction(todproc)
        todproc.triggered.connect(self.tod_func_processing)

        self.pointingmenu = menubar.addMenu('Pointing Functions')
        lstlatfunc = QAction('LAT and LST paramters',  self)
        self.pointingmenu.addAction(lstlatfunc)
        lstlatfunc.triggered.connect(self.lstlat_func_offset)
        refpoint = QAction('Reference Point', self)
        self.pointingmenu.addAction(refpoint)
        refpoint.triggered.connect(self.refpoint_func)

    def current_name(self):
        self.experiment_name = self.TabLayout.experiment_name

    @pyqtSlot()
    def lstlat_func_offset(self):
        dialog = LST_LAT_Param(experiment = self.experiment_name)
        dialog.LSTtype_signal.connect(self.connection_LST_type)
        dialog.LATtype_signal.connect(self.connection_LAT_type)
        dialog.LSTconv_signal.connect(self.connection_LST_conv)
        dialog.LATconv_signal.connect(self.connection_LAT_conv)
        dialog.lstlatfreqsignal.connect(self.connection_LSTLAT_freq)
        dialog.lstlatsamplesignal.connect(self.connection_LSTLAT_sample)
        dialog.exec_()
    
    @pyqtSlot()
    def refpoint_func(self):
        dialog = REFPOINT_Param()
        dialog.radecsignal.connect(self.connection_ref_point)
        dialog.exec_()

    @pyqtSlot()
    def tod_func_offset(self):
        dialog = TODoffsetWindow()
        
        try:
            if self.TabLayout.todoffsetvalue is not None:
                dialog.todoffsetvalue.setText(str(self.TabLayout.todoffsetvalue))
            else:
                dialog.todoffsetvalue.setText('0.0')
        except AttributeError:
            pass
        dialog.todoffsetsignal.connect(self.connection_tod_off)
        dialog.exec_()

    @pyqtSlot()
    def tod_func_processing(self):
        dialog = TOD_processing()
        try:
            if self.TabLayout.todpolynomialvalue is not None:
                dialog.polynomialvalue.setText(str(self.TabLayout.todpolynomialvalue))
            else:
                dialog.polynomialvalue.setText('5')
        except AttributeError:
            pass

        try:
            if self.TabLayout.sigmavalue is not None:
                dialog.sigmavalue.setText(str(self.TabLayout.todsigmavalue))
            else:
                dialog.sigmavalue.setText('5')
        except AttributeError:
            pass

        try:
            if self.TabLayout.todprominencevalue is not None:
                dialog.prominencevalue.setText(str(self.TabLayout.todprominencevalue))
            else:
                dialog.prominencevalue.setText('5')
        except AttributeError:
            pass
        
        dialog.polynomialordersignal.connect(self.connection_tod_polynomial)
        dialog.sigmasignal.connect(self.connection_tod_sigma)
        dialog.prominencesignal.connect(self.connection_tod_prominence)
        dialog.despikesignal.connect(self.connection_tod_despike)
        dialog.exec_()

    @pyqtSlot()
    def beam_fit_param_menu(self):
        dialog = BeamFitParamWindow()
        dialog.fitparamsignal.connect(self.connection_beam_param)
        dialog.exec_()

    @pyqtSlot(np.ndarray)
    def connection_beam_param(self, val):
        self.TabLayout.beamparam = val.copy()
    
    @pyqtSlot(float)
    def connection_tod_off(self, val):
        self.TabLayout.todoffsetvalue = copy.copy(val)

    @pyqtSlot(int)
    def connection_tod_polynomial(self, val):
        self.TabLayout.todpolynomialvalue = copy.copy(val)

    @pyqtSlot(int)
    def connection_tod_sigma(self, val):
        self.TabLayout.todsigmavalue = copy.copy(val)

    @pyqtSlot(int)
    def connection_tod_prominence(self, val):
        self.TabLayout.todprominencevalue = copy.copy(val)

    @pyqtSlot(bool)
    def connection_tod_despike(self, val):
        self.TabLayout.toddespikevalue = copy.copy(val)

    @pyqtSlot(str)
    def connection_LST_type(self, val):
        self.TabLayout.LSTtype = copy.copy(val)
    
    @pyqtSlot(str)
    def connection_LAT_type(self, val):
        self.TabLayout.LATtype = copy.copy(val)

    @pyqtSlot(np.ndarray)
    def connection_LST_conv(self, val):
        self.TabLayout.LSTconv = val.copy()

    @pyqtSlot(np.ndarray)
    def connection_LAT_conv(self, val):
        self.TabLayout.LATconv = val.copy()

    @pyqtSlot(float)
    def connection_LSTLAT_freq(self, val):
        self.TabLayout.lstlatfreq = copy.copy(val)

    @pyqtSlot(float)
    def connection_LSTLAT_sample(self, val):
        self.TabLayout.lstlatsampleframe = copy.copy(val)

    @pyqtSlot(np.ndarray)
    def connection_ref_point(self, val):
        self.TabLayout.refpoint = val.copy()    

    def closeEvent(self,event):

        '''
        This function contains the code that is run when the application is closed.
        In this case, deleting the pickles file created.
        '''

        result = QMessageBox.question(self,
                                      "Confirm Exit...",
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes:
            
            directory = 'pickles_object/'
            pkl_list = os.listdir(directory)

            if np.size(pkl_list) > 0:
                for i in range(len(pkl_list)):
                    path = directory+pkl_list[i]
                    os.remove(path)

            event.accept()

class LST_LAT_Param(QDialog):

    LSTtype_signal = pyqtSignal(str)
    LATtype_signal = pyqtSignal(str)
    LSTconv_signal = pyqtSignal(np.ndarray)
    LATconv_signal = pyqtSignal(np.ndarray)
    lstlatfreqsignal = pyqtSignal(float)
    lstlatsamplesignal = pyqtSignal(float)

    def __init__(self, experiment, parent = None):
        super(QDialog, self).__init__(parent)
        
        self.setWindowTitle('LAT and LST Parameters')

        self.LSTtype = QLineEdit('')
        self.LSTlabel = QLabel('LST File Type')

        self.LATtype = QLineEdit('')
        self.LATlabel = QLabel('LAT File Type')

        self.aLSTconv = QLineEdit('')
        self.bLSTconv = QLineEdit('')
        self.LSTconv = QLabel('LST DIRFILE conversion factors')
        self.LSTconv.setBuddy(self.aLSTconv)

        self.aLATconv = QLineEdit('')
        self.bLATconv = QLineEdit('')
        self.LATconv = QLabel('LAT DIRFILE conversion factors')
        self.LATconv.setBuddy(self.aLATconv)

        self.LSTLATfreq = QLineEdit('')
        self.LSTLATfreqlabel = QLabel('LST/LAT frequency Sample')

        self.LSTLATsample = QLineEdit('')
        self.LSTLATsamplelabel = QLabel('LST/LAT Samples per frame')

        self.savebutton = QPushButton('Write Parameters')
        self.savebutton.clicked.connect(self.updateParamValues)

        layout = QGridLayout(self)

        layout.addWidget(self.LSTlabel, 0, 0)
        layout.addWidget(self.LSTtype, 0, 1)
        layout.addWidget(self.LATlabel, 1, 0)
        layout.addWidget(self.LATtype, 1, 1)
        layout.addWidget(self.LSTconv, 2, 0)
        layout.addWidget(self.aLSTconv, 2, 1)
        layout.addWidget(self.bLSTconv, 2, 2)
        layout.addWidget(self.LATconv, 3, 0)
        layout.addWidget(self.aLATconv, 3, 1)
        layout.addWidget(self.bLATconv, 3, 2)
        layout.addWidget(self.LSTLATfreqlabel, 4, 0)
        layout.addWidget(self.LSTLATfreq, 4, 1)
        layout.addWidget(self.LSTLATsamplelabel, 5, 0)
        layout.addWidget(self.LSTLATsample, 5, 1)

        layout.addWidget(self.savebutton)

        self.configuration_value(experiment=experiment)

        self.setLayout(layout)

    def configuration_value(self, experiment):
    
        dir_path = os.getcwd()+'/config/'
        
        filepath = dir_path+experiment.lower()+'.cfg'
        model = configparser.ConfigParser()

        model.read(filepath)
        sections = model.sections()

        for section in sections:
            if section.lower() == 'lst_lat parameters':
                lstlatfreq_config = float(model.get(section, 'LSTLATFREQ').split('#')[0])
                lst_dir_conv = model.get(section,'LST_DIR_CONV').split('#')[0].strip()
                lstconv_config = np.array(lst_dir_conv.split(',')).astype(float)
                lat_dir_conv = model.get(section,'LAT_DIR_CONV').split('#')[0].strip()
                latconv_config = np.array(lat_dir_conv.split(',')).astype(float)
                lstlatframe_config = float(model.get(section, 'LSTLAT_SAMP_FRAME').split('#')[0])
                lsttype_config = model.get(section,'LST_FILE_TYPE').split('#')[0].strip()
                lattype_config = model.get(section,'LAT_FILE_TYPE').split('#')[0].strip()

        self.LSTLATfreq.setText(str(lstlatfreq_config))
        self.LSTLATsample.setText(str(lstlatframe_config))
        self.LSTtype.setText(str(lsttype_config))
        self.LATtype.setText(str(lattype_config))
        self.aLSTconv.setText(str(lstconv_config[0]))
        self.bLSTconv.setText(str(lstconv_config[1]))
        self.aLATconv.setText(str(latconv_config[0]))
        self.bLATconv.setText(str(latconv_config[1]))

    def updateParamValues(self):

        self.LSTtype_signal.emit(self.LSTtype.text())
        self.LATtype_signal.emit(self.LATtype.text())

        LST_array = np.array([float(self.aLSTconv.text()), \
                              float(self.bLSTconv.text())])

        LAT_array = np.array([float(self.aLATconv.text()), \
                              float(self.bLATconv.text())])

        self.LSTconv_signal.emit(LST_array)
        self.LATconv_signal.emit(LAT_array)
        self.lstlatfreqsignal.emit(float(self.LSTLATfreq.text()))
        self.lstlatsamplesignal.emit(float(self.LSTLATsample.text()))

        self.close()

class REFPOINT_Param(QDialog):

    '''
    Class to create a dialog input for the coordinates of the reference point.
    The reference point is used to compute the pointing offset
    '''

    radecsignal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('Reference Point for pointing offset')

        self.ra = QLineEdit('')
        self.ralabel = QLabel('RA in degree')
        self.dec = QLineEdit('')
        self.declabel = QLabel('DEC in degree')

        self.savebutton = QPushButton('Write Parameters')
        self.savebutton.clicked.connect(self.updateParamValues)

        self.layout = QGridLayout()

        self.layout.addWidget(self.ralabel, 0, 0)
        self.layout.addWidget(self.ra, 0, 1)
        self.layout.addWidget(self.declabel, 1, 0)
        self.layout.addWidget(self.dec, 1, 1)
        self.layout.addWidget(self.savebutton)

        self.setLayout(self.layout)

    def updateParamValues(self):

        self.ra_value = float(self.ra.text())
        self.dec_value = float(self.dec.text())

        self.radecsignal.emit(np.array([self.ra_value,self.dec_value]))
        self.close() 

class BeamFitParamWindow(QDialog):

    '''
    Dialog window for adding manually beam fitting parameters
    '''

    fitparamsignal = pyqtSignal(np.ndarray)

    def __init__(self, parent = None):
        super(QDialog, self).__init__(parent)
        
        self.setWindowTitle('Beam Fitting Parameters')

        self.peak_numbers = QLineEdit('')
        self.peak_numbers_label = QLabel('Number of Gaussin to be fitted')

        self.savebutton = QPushButton('Write Parameters')

        self.table = QTableWidget()

        self.peak_numbers.textChanged[str].connect(self.updateTable)
        self.savebutton.clicked.connect(self.updateParamValues)

        layout = QGridLayout(self)

        layout.addWidget(self.peak_numbers_label, 0, 0)
        layout.addWidget(self.peak_numbers, 0, 1)
        layout.addWidget(self.table, 1, 0, 1, 2)
        layout.addWidget(self.savebutton)

        self.setLayout(layout)

        self.resize(640, 480)
        
    def configureTable(self, table, rows):
        table.setColumnCount(6)
        table.setRowCount(rows)
        table.setHorizontalHeaderItem(0, QTableWidgetItem("Amplitude"))
        table.setHorizontalHeaderItem(1, QTableWidgetItem("X0 (in pixel)"))
        table.setHorizontalHeaderItem(2, QTableWidgetItem("Y0 (in pixel)"))
        table.setHorizontalHeaderItem(3, QTableWidgetItem("SigmaX (in pixel)"))
        table.setHorizontalHeaderItem(4, QTableWidgetItem("SigmaY (in pixel)"))
        table.setHorizontalHeaderItem(5, QTableWidgetItem("Theta"))
        #table.horizontalHeader().setStretchLastSection(True)

    def updateTable(self):
        try:
            rows = int(self.peak_numbers.text().strip())
            self.configureTable(self.table, rows)
        except ValueError:
            pass

    def updateParamValues(self):

        values = np.array([])
        rows_number = self.table.rowCount()
        column_number = self.table.columnCount()

        for i in range(rows_number):
            for j in range(column_number):
                values = np.append(values, float(self.table.item(i,j).text()))

        self.fitparam = values.copy()
        self.fitparamsignal.emit(values)

class ProgressBarWindow(QDialog):

    def __init__(self, parent = None):
        super(ProgressBarWindow, self).__init__(parent)
        self.setGeometry(150, 150, 300, 100)
        self.setWindowTitle('Current Progress')

        self.progress = QProgressBar()
        
        self.strstatus = 'Current Status   '
        self.currentstatus = QLabel(self.strstatus)

        layout = QGridLayout(self)

        layout.addWidget(self.currentstatus)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.show()

    def setValue(self, val): 
        self.progress.setValue(val)

    def setCurrentAction(self, action):
        string = self.strstatus+action
        self.currentstatus.setText(string)

class TODoffsetWindow(QDialog):

    '''
    Dialog window for adding manually time offset between the 
    detector TOD and the pointing solution
    '''

    todoffsetsignal = pyqtSignal(float)

    def __init__(self, parent = None):
        super(QDialog, self).__init__(parent)
        #w = QDialog(self)

        self.setWindowTitle('Detectors TOD timing offset')

        self.todoffsetvalue = QLineEdit('')
        self.todoffset_label = QLabel('TOD timing offset value (ms)')

        self.savebutton = QPushButton('Write Parameters')

        layout = QGridLayout(self)

        layout.addWidget(self.todoffset_label, 0, 0)
        layout.addWidget(self.todoffsetvalue, 0, 1)
        layout.addWidget(self.savebutton)

        self.setLayout(layout)

        self.savebutton.clicked.connect(self.updateParamValues)

    def updateParamValues(self):

        self.value = float(self.todoffsetvalue.text())

        self.todoffsetsignal.emit(self.value)
        self.close()

class TOD_processing(QDialog):

    polynomialordersignal = pyqtSignal(int)
    sigmasignal = pyqtSignal(int)
    prominencesignal = pyqtSignal(int)
    despikesignal = pyqtSignal(bool)

    def __init__(self, parent = None):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('TOD Processing Parameters')

        self.polynomialorder = QLineEdit('')
        self.polynomialorderlabel = QLabel('Order of the trend polynomial to be removed. If 0 no removal is performed')

        self.despikebox = QCheckBox('Apply Despiking')

        self.sigma = QLineEdit('')
        self.sigmalabel = QLabel('Height of a peak in unit of sigma')

        self.prominence = QLineEdit('')
        self.prominencelabel = QLabel('Prominence of a peak in unit of sigma')

        self.savebutton = QPushButton('Write Parameters')

        layout = QGridLayout(self)

        layout.addWidget(self.polynomialorderlabel, 0, 0)
        layout.addWidget(self.polynomialorder, 0, 1)
        layout.addWidget(self.despikebox, 1, 0)
        layout.addWidget(self.sigmalabel, 2, 0)
        layout.addWidget(self.sigma, 2, 1)
        layout.addWidget(self.prominencelabel, 3, 0)
        layout.addWidget(self.prominence, 3, 1)
        layout.addWidget(self.savebutton)

        self.setLayout(layout)

        self.savebutton.clicked.connect(self.updateParamValues)

    def updateParamValues(self):
        
        try:
            self.polynomialvalue = int(self.polynomialorder.text())
        except ValueError:
            self.polynomialvalue = 5

        try:
            self.sigmavalue = int(self.sigma.text())
        except ValueError:
            self.sigmavalue = 5

        try:
            self.prominencevalue = int(self.prominence.text())
        except ValueError:
            self.prominencevalue = 5
        
        self.despikebool = self.despikebox.isChecked()

        self.despikesignal.emit(self.despikebool)
        self.prominencesignal.emit(self.prominencevalue)
        self.polynomialordersignal.emit(self.polynomialvalue)
        self.sigmasignal.emit(self.sigmavalue)
        self.close()

class MainWindowTab(QTabWidget):

    '''
    General layout of the application 
    '''

    def __init__(self, parent = None):
        super(MainWindowTab, self).__init__(parent)
        self.tab1 = ParamMapTab()
        self.tab2 = TODTab()

        self.addTab(self.tab1,"Parameters and Maps")
        self.addTab(self.tab2,"Detector TOD")

        checkI = self.tab1.ICheckBox

        self.emit_name()
        self.tab1.experiment.activated[str].connect(self.emit_name)
        
        self.data = np.array([])
        self.cleandata = np.array([])

        self.beamparam = None
        self.todoffsetvalue = None 
        self.todpolynomialvalue = 5
        self.todsigmavalue = 5
        self.todprominencevalue = 5
        self.toddespikevalue = True

        #LAT and LST parameters
        self.LSTtype = None
        self.LATtype = None
        self.LSTconv = np.array([1., 0.])
        self.LATconv = np.array([1., 0.])
        self.lstlatfreq = None
        self.lstlatsampleframe = None

        self.refpoint = None

        self.tab3 = BeamTab(checkbox=checkI, ctype=self.tab1.coordchoice.currentText())
        self.addTab(self.tab3, "Beam")

        self.tab1.plotbutton.clicked.connect(self.updatedata)
        self.tab1.fitsbutton.clicked.connect(self.save2fits)

        self.tab2.detcombolist.activated[str].connect(self.drawdetTOD)

    def emit_name(self):

        self.experiment_name = (self.tab1.experiment.currentText())

    def updatedata(self):
        '''
        This function updates the map values everytime that the plot button is pushed
        '''

        if self.tab1.PointingOffsetCheckBox.isChecked() or self.tab1.DettableCheckBox.isChecked():
            correction = True
        else:
            correction = False
        pb =  ProgressBarWindow()
        pb.setCurrentAction('Loading Data')
        pb.setValue(0)
        #functions to compute the updated values

        self.tab1.load_func(offset = self.todoffsetvalue, correction = correction, \
                            LSTtype=self.LSTtype, LATtype=self.LATtype,\
                            LSTconv=self.LSTconv, LATconv=self.LATconv, \
                            lstlatfreq=self.lstlatfreq, lstlatsample = self.lstlatsampleframe, \
                            polynomialorder = int(self.todpolynomialvalue), despike = self.toddespikevalue, \
                            sigma = int(self.todsigmavalue), prominence=int(self.todprominencevalue))

        self.data = self.tab1.detslice
        self.lst = self.tab1.lstslice
        self.lat = self.tab1.latslice


        pb.setCurrentAction('Processing Data')
        pb.setValue(25)

        self.cleandata = self.tab1.cleaned_data

        self.tab2.detlist_selection(self.tab1.det_list)

        pb.setCurrentAction('Drawing TOD')
        pb.setValue(50)

        if np.size(np.shape(self.data)) == 1:
            self.tab2.draw_TOD(self.data)
            self.tab2.draw_cleaned_TOD(self.cleandata)
        else:
            self.tab2.draw_TOD(self.data[0])
            self.tab2.draw_cleaned_TOD(self.cleandata[0])

        pb.setCurrentAction('Drawing Maps')
        pb.setValue(75)

        try:
            
            self.tab1.mapvalues(self.cleandata)
            
            #Update Maps
            maps = self.tab1.map_value
            mp_ini = self.tab1.createMapPlotGroup
            
            if np.size(self.tab1.det_list) == 1:
                x_min_map = np.floor(np.amin(self.tab1.w[:,0]))
                # x_max_map = np.floor(np.amax(self.tab1.w[:,0])) 
                y_min_map = np.floor(np.amin(self.tab1.w[:,1]))
                # y_max_map = np.floor(np.amax(self.tab1.w[:,1]))
                index1, = np.where(self.tab1.w[:,0]<0)
                index2, = np.where(self.tab1.w[:,1]<0)
            else:
                x_min_map = np.floor(np.amin(self.tab1.w[:,:,0]))
                # x_max_map = np.floor(np.amax(self.tab1.w[:,:,0])) 
                y_min_map = np.floor(np.amin(self.tab1.w[:,:,1]))
                # y_max_map = np.floor(np.amax(self.tab1.w[:,:,1]))
                index1, = np.where(self.tab1.w[0,:,0]<0)
                index2, = np.where(self.tab1.w[0,:,1]<0)
 
            if np.size(index1) > 1:
                crpix1_new  = (self.tab1.crpix[0]-x_min_map)
            else:
                crpix1_new = copy.copy(self.tab1.crpix[0])
            
            if np.size(index2) > 1:
                crpix2_new  = (self.tab1.crpix[1]-y_min_map)
            else:
                crpix2_new = copy.copy(self.tab1.crpix[1])

            crpix_new = np.array([crpix1_new, crpix2_new])

            wcsworld = mp.wcs_world(self.tab1.ctype, crpix_new, self.tab1.cdelt, self.tab1.crval)

            coord_test, self.proj_new = wcsworld.world(np.reshape(self.tab1.crval, (1,2)), self.tab1.parallactic)
            # if crpix_new[0]*2 < x_max_map:
            #     x_sel = np.array([crpix_new[0]-self.tab1.pixnum[0]/2, crpix_new[0]+self.tab1.pixnum[0]/2], dtype=int)
            # else:
            #     x_sel = np.array([x_max_map-self.tab1.pixnum[0],x_max_map], dtype=int)

            # if crpix_new[1]*2 < y_max_map:
            #     y_sel = np.array([crpix_new[1]-self.tab1.pixnum[1]/2, crpix_new[1]+self.tab1.pixnum[1]/2], dtype=int)
            # else:
            #     y_sel = np.array([y_max_map-self.tab1.pixnum[1],y_max_map], dtype=int)

            if self.tab1.coordchoice.currentText() == 'XY Stage':
                xystagebool = True
            else:
                xystagebool = False
            
            ctype = self.tab1.coordchoice.currentText()
            print('CTYPE_1', ctype)
            mp_ini.updateTab(data=maps, coord1 = self.tab1.coord1slice, coord2 = self.tab1.coord2slice, \
                             crval = self.tab1.crval, ctype = ctype, pixnum = self.tab1.pixnum, \
                             telcoord = self.tab1.telescopecoordinateCheckBox.isChecked(),\
                             crpix = crpix_new, cdelt = self.tab1.cdelt, projection = self.proj_new, xystage=xystagebool)
            # cutout = mp_ini.cutout

            # self.proj_new = cutout.wcs
            

            #Update Offset
            if self.tab1.PointingOffsetCalculationCheckBox.isChecked():
                if self.refpoint is not None:
                    self.tab1.updateOffsetValue(self.refpoint[0], self.refpoint[1])
                else:
                    self.tab1.updateOffsetValue()


            '''
            checkBeam = self.tab1.BeamCheckBox

            #Create Beams
            if checkBeam.isChecked():
                print('BEAM')
                beam_value = bm.beam(maps, param = self.beamparam)
                beam_map = beam_value.beam_fit()

                param = beam_map[1]

                beams = self.tab3.beammaps

                if isinstance(beam_map[0], str):
                    self.warningbox = QMessageBox()
                    self.warningbox.setIcon(QMessageBox.Warning)
                    self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                    self.warningbox.setWindowTitle('Warning')

                    msg = 'Fit not converged'
                    
                    self.warningbox.setText(msg)        
                
                    self.warningbox.exec_()

                else:
                    beams.updateTab(data=beam_map[0], coord1 = self.tab1.coord1slice, coord2 = self.tab1.coord2slice, \
                                    crval = self.tab1.crval, ctype=ctype, pixnum = self.tab1.pixnum, telcoord = self.tab1.telescopecoordinateCheckBox.isChecked(),\
                                    crpix = crpix_new, cdelt = self.tab1.cdelt, projection = self.proj_new, xystage=xystagebool)
                    self.tab3.updateTable(param)
            '''
        except AttributeError:
            pass
        
        pb.close()

    def drawdetTOD(self):
        
        index = self.tab2.detcombolist.currentIndex()

        if np.size(np.shape(self.data)) == 1:
            self.tab2.draw_TOD(self.data)
            self.tab2.draw_cleaned_TOD(self.cleandata)
        else:
            print(self.data[index])
            self.tab2.draw_TOD(self.data[index])
            self.tab2.draw_cleaned_TOD(self.cleandata[index])

    def save2fits(self): #function to save the map as a FITS file
        hdr = self.tab1.proj.to_header() #grabs the projection information for header
        maps = self.tab1.map_value #grabs the actual map for the fits img
        hdu = fits.PrimaryHDU(maps, header = hdr)
        hdu.writeto('./'+self.tab1.fitsname.text())

class ParamMapTab(QWidget):

    '''
    Create the layout of the first tab containing the various input parameters and 
    the final maps
    '''

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.detslice = np.array([])         #Detector TOD between the frames of interest
        self.latslice = np.array([])
        self.lstslice = np.array([])
        self.cleaned_data = np.array([])     #Detector TOD cleaned (despiked and highpassed) between the frame of interest
        self.proj = None                     #WCS projection of the map
        self.map_value = np.array([])        #Final map values

        self.experiment = QComboBox()
        self.experiment.addItem('BLAST-TNG')
        self.experiment.addItem('BLASTPol')
        self.experimentLabel = QLabel("Experiment:")
        self.experimentLabel.setBuddy(self.experiment)

        self.createAstroGroup()
        self.createExperimentGroup()
        self.createDataRepository()
        
        self.plotbutton = QPushButton('Plot')
        self.fitsbutton = QPushButton('Save as Fits')

        self.createOffsetGroup()
        mainlayout = QGridLayout(self)

        self.createMapPlotGroup = MapPlotsGroup(checkbox=self.ICheckBox, data=self.map_value)

        self.fitsname = QLineEdit('')
        self.fitsnamelabel = QLabel("FITS name")
        self.fitsnamelabel.setBuddy(self.fitsname)

        scroll2 = QScrollArea()
        scroll2.setWidget(self.ExperimentGroup)
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(200)

        ExperimentGroup_Scroll = QGroupBox("Experiment Parameters")
        ExperimentGroup_Scroll.setFlat(True)
        ExperimentGroup_Scroll.setLayout(QVBoxLayout())
        ExperimentGroup_Scroll.layout().addWidget(scroll2)
        mainlayout.addWidget(self.experimentLabel, 0, 0)
        mainlayout.addWidget(self.experiment, 0, 1)
        mainlayout.addWidget(self.DataRepository, 1, 0, 1, 2)
        mainlayout.addWidget(self.AstroGroup, 2, 0, 1, 2)
        mainlayout.addWidget(scroll2, 3, 0, 1, 2)
        mainlayout.addWidget(self.plotbutton, 4, 0, 1, 2)
        mainlayout.addWidget(self.createMapPlotGroup, 0, 2, 3, 2)
        mainlayout.addWidget(self.OffsetGroup, 3, 2)
        mainlayout.addWidget(self.fitsbutton,4,2)
        mainlayout.addWidget(self.fitsname)
        mainlayout.addWidget(self.fitsnamelabel)
        
        self.setLayout(mainlayout)        

    def createDataRepository(self):

        '''
        Function for the layout and input of the Data Repository group.
        This includes:
        - Paths to the data
        - Name of the detectors
        - Possibility to use pointing offset and detectortables
        - Roach Number
        '''

        self.DataRepository = QGroupBox("Data Repository")
        
        #self.detpath = QLineEdit('/mnt/c/Users/gabri/Documents/GitHub/mapmaking/2012_data/bolo_data/')
        self.detpath = QLineEdit('/home/mvancuyck/Desktop/master')
        #self.detpath = QLineEdit('/mnt/d/data/etc/mole.lnk')
        self.detpathlabel = QLabel("Detector Path:")
        self.detpathlabel.setBuddy(self.detpath)

        self.detname = QLineEdit('4')
        self.detnamelabel = QLabel("Detector Name:")
        self.detnamelabel.setBuddy(self.detname)

        self.detvalue = np.array([])

        self.roachnumber = QLineEdit('3')
        self.roachnumberlabel = QLabel("Roach Number:")
        self.roachnumberlabel.setBuddy(self.roachnumber)

        #self.coordpath = QLineEdit('/mnt/c/Users/gabri/Documents/GitHub/mapmaking/2012_data/')
        self.coordpath = QLineEdit('/home/mvancuyck/Desktop/master')
        #self.coordpath = QLineEdit('/mnt/d/data/etc/mole.lnk')
        self.coordpathlabel = QLabel("Coordinate Path:")
        self.coordpathlabel.setBuddy(self.coordpath)

        self.coord1value = np.array([])
        self.coord2value = np.array([])

        self.DettableCheckBox = QCheckBox("Use Detector Table")
        self.DettableCheckBox.setChecked(False)
        self.dettablepath = QLineEdit('/mnt/c/Users/gabri/Documents/GitHub/mapmaker/')
        self.dettablepathlabel = QLabel("Detector Table Path:")
        self.dettablepathlabel.setBuddy(self.dettablepath)

        self.DettableCheckBox.toggled.connect(self.dettablepathlabel.setVisible)
        self.DettableCheckBox.toggled.connect(self.dettablepath.setVisible)

        self.PointingOffsetCheckBox = QCheckBox("Use Pointing Offset")
        self.PointingOffsetCheckBox.setChecked(False)
        self.pointingoffsetnumber = QLineEdit('1')
        self.pointingoffsetnumberlabel = QLabel("StarCamera used for pointing offset:")
        self.pointingoffsetnumberlabel.setBuddy(self.pointingoffsetnumber)

        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetnumberlabel.setVisible)
        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetnumber.setVisible)

        self.layout = QGridLayout()

        self.layout.addWidget(self.detpathlabel, 0, 0)
        self.layout.addWidget(self.detpath, 0, 1, 1, 2)
        self.layout.addWidget(self.detnamelabel, 1, 0)
        self.layout.addWidget(self.detname, 1, 1, 1, 2)
        self.layout.addWidget(self.roachnumberlabel, 2, 0)
        self.layout.addWidget(self.roachnumber, 2, 1, 1, 2)
        self.layout.addWidget(self.coordpathlabel, 3, 0)
        self.layout.addWidget(self.coordpath, 3, 1, 1, 2)
        self.layout.addWidget(self.DettableCheckBox, 4, 0)
        self.layout.addWidget(self.dettablepathlabel, 5, 1)
        self.layout.addWidget(self.dettablepath, 5, 2)
        self.layout.addWidget(self.PointingOffsetCheckBox, 6, 0)
        self.layout.addWidget(self.pointingoffsetnumberlabel, 7, 1)
        self.layout.addWidget(self.pointingoffsetnumber, 7, 2)

        self.dettablepathlabel.setVisible(False)
        self.dettablepath.setVisible(False)
        self.pointingoffsetnumberlabel.setVisible(False)
        self.pointingoffsetnumber.setVisible(False)

        self.DataRepository.setLayout(self.layout)

    def createAstroGroup(self):

        '''
        Function for the layout and input of the Astronometry parameters group.
        This includes:
        - Coordinates system
        - Standard WCS parameters to create a map
        - If the maps need to be convolved
        '''

        self.AstroGroup = QGroupBox("Astronomy Parameters")
    
        self.coordchoice = QComboBox()
        self.coordchoice.addItem('RA and DEC')
        self.coordchoice.addItem('AZ and EL')
        self.coordchoice.addItem('CROSS-EL and EL')
        self.coordchoice.addItem('XY Stage')
        coordLabel = QLabel("Coordinates System:")
        coordLabel.setBuddy(self.coordchoice)
        self.telescopecoordinateCheckBox = QCheckBox('Use Telescope Coordinates')

        self.convchoice = QComboBox()
        self.convchoice.addItem('Not Apply')
        self.convchoice.addItem('Gaussian')
        convLabel = QLabel("Map Convolution:")
        convLabel.setBuddy(self.convchoice)

        self.GaussianSTD = QLineEdit('')
        self.gaussianLabel = QLabel("Convolution STD (in arcsec):")
        self.gaussianLabel.setBuddy(self.GaussianSTD)

        self.crpix1 = QLineEdit('50')
        self.crpix2 = QLineEdit('50')
        self.crpixlabel = QLabel("CRpix of the Map:")
        self.crpixlabel.setBuddy(self.crpix1)

        self.cdelt1 = QLineEdit('0.1')
        self.cdelt2 = QLineEdit('0.1')
        self.cdeltlabel = QLabel("Cdelt of the Map in deg:")
        self.cdeltlabel.setBuddy(self.cdelt1)

        self.crval1 = QLineEdit('230.0')
        self.crval2 = QLineEdit('-55.79')
        self.crvallabel = QLabel("Cval of the Map in deg:")
        self.crvallabel.setBuddy(self.crval1)

        self.pixnum1 = QLineEdit('100')
        self.pixnum2 = QLineEdit('100')
        self.pixnumlabel = QLabel("Pixel Number:")
        self.pixnumlabel.setBuddy(self.pixnum1)

        self.ICheckBox = QCheckBox("Map only I")
        self.ICheckBox.setChecked(True)

        self.BeamCheckBox = QCheckBox("Beam Analysis")
        self.BeamCheckBox.setChecked(False)

        self.PointingOffsetCalculationCheckBox = QCheckBox("Calculate Pointing Offset")
        self.PointingOffsetCalculationCheckBox.setChecked(False)
        
        self.convchoice.activated[str].connect(self.updateGaussian)
        self.coordchoice.activated[str].connect(self.configuration_update)
           
        layout = QGridLayout()
        layout.addWidget(coordLabel, 0, 0)
        layout.addWidget(self.coordchoice, 0, 1, 1, 2)
        layout.addWidget(self.telescopecoordinateCheckBox, 1, 0)
        layout.addWidget(convLabel, 2, 0)
        layout.addWidget(self.convchoice, 2, 1, 1, 2)
        layout.addWidget(self.gaussianLabel, 3, 1)
        layout.addWidget(self.GaussianSTD, 3, 2)
        layout.addWidget(self.crpixlabel, 4, 0)
        layout.addWidget(self.crpix1, 4, 1)
        layout.addWidget(self.crpix2, 4, 2)
        layout.addWidget(self.cdeltlabel, 5, 0)
        layout.addWidget(self.cdelt1, 5, 1)
        layout.addWidget(self.cdelt2, 5, 2)
        layout.addWidget(self.crvallabel, 6, 0)
        layout.addWidget(self.crval1, 6, 1)
        layout.addWidget(self.crval2, 6, 2)
        layout.addWidget(self.pixnumlabel, 7, 0)
        layout.addWidget(self.pixnum1, 7, 1)
        layout.addWidget(self.pixnum2, 7, 2)
        layout.addWidget(self.ICheckBox, 8, 0)
        layout.addWidget(self.BeamCheckBox, 9, 0)
        layout.addWidget(self.PointingOffsetCalculationCheckBox, 10, 0)

        self.GaussianSTD.setVisible(False)
        self.gaussianLabel.setVisible(False)
        
        self.AstroGroup.setLayout(layout)

    def updateGaussian(self, text=None):

        '''
        Function to update the layout of the group, to add a line 
        to input the std of the gaussian convolution if the convolution parameter
        is set to gaussian
        '''

        if text is None:
            text = self.convchoice.currentText()

        if text == 'Gaussian': 
            self.GaussianSTD.setVisible(True)
            self.GaussianSTD.setEnabled(True)
            self.gaussianLabel.setVisible(True)
            self.gaussianLabel.setEnabled(True)
        else: 
            self.GaussianSTD.setVisible(False)
            self.GaussianSTD.setEnabled(False)
            self.gaussianLabel.setVisible(False)
            self.gaussianLabel.setEnabled(False)

    def createExperimentGroup(self):

        '''
        Function for the layout and input of the Experiment parameters group.
        This includes:
        - Frequency sampling of detectors and ACSs 
        - Experiment to be analyzed 
        - Frames of interests
        - High pass filter cutoff frequency
        - If DIRFILE conversion needs to be performed. If so, the parameters to 
          use for the conversion
        '''

        self.ExperimentGroup = QGroupBox()
        self.ExperimentGroup.setFlat(True)

        self.detfreq = QLineEdit('')
        self.detfreqlabel = QLabel("Detector Frequency Sample")
        self.detfreqlabel.setBuddy(self.detfreq)
        self.acsfreq = QLineEdit('')
        self.acsfreqlabel = QLabel("ACS Frequency Sample")
        self.acsfreqlabel.setBuddy(self.acsfreq)

        self.highpassfreq = QLineEdit('0.1')
        self.highpassfreqlabel = QLabel("High Pass Filter cutoff frequency")
        self.highpassfreqlabel.setBuddy(self.highpassfreq)

        self.detframe = QLineEdit('')
        self.detframelabel = QLabel("Detector Samples per Frame")        
        self.detframelabel.setBuddy(self.detframe)
        self.acsframe = QLineEdit('')
        self.acsframelabel = QLabel("ACS Sample Samples per Frame")
        self.acsframelabel.setBuddy(self.acsframe)

        self.startframe = QLineEdit('72373')
        self.endframe = QLineEdit('79556')
        self.numberframelabel = QLabel('Starting and Ending Frames')
        self.numberframelabel.setBuddy(self.startframe)

        self.dettype = QLineEdit('')      
        self.dettypelabel = QLabel("Detector DIRFILE data type")
        self.dettypelabel.setBuddy(self.dettype)
        self.coord1type = QLineEdit('')
        self.coord1typelabel = QLabel("Coordinate 1 DIRFILE data type")
        self.coord1typelabel.setBuddy(self.coord1type)
        self.coord2type = QLineEdit('')
        self.coord2typelabel = QLabel("Coordinate 2 DIRFILE data type")
        self.coord2typelabel.setBuddy(self.coord2type)

        self.DirConvCheckBox = QCheckBox("DIRFILE Conversion factors")
        self.DirConvCheckBox.setChecked(False)

        self.adetconv = QLineEdit('')
        self.bdetconv = QLineEdit('')
        self.detconv = QLabel('Det conversion factors')
        self.detconv.setBuddy(self.adetconv)

        self.acoord1conv = QLineEdit('')
        self.bcoord1conv = QLineEdit('')
        self.coord1conv = QLabel('Coord 1 conversion factors')
        self.coord1conv.setBuddy(self.acoord1conv)

        self.acoord2conv = QLineEdit('')
        self.bcoord2conv = QLineEdit('')
        self.coord2conv = QLabel('Coord 2 conversion factors')
        self.coord2conv.setBuddy(self.acoord2conv)

        self.HWPCheckBox = QCheckBox("USE HWP")
        self.HWPCheckBox.setChecked(False)

        self.HWPtype = QLineEdit('')      
        self.HWPtypelabel = QLabel("HWP DIRFILE data type")
        self.HWPtypelabel.setBuddy(self.HWPtype)

        self.HWPfreq = QLineEdit('')
        self.HWPfreqlabel = QLabel("HWP Frequency Sample")
        self.HWPfreqlabel.setBuddy(self.HWPfreq)

        self.HWPframe = QLineEdit('')
        self.HWPframelabel = QLabel("HWP Sample Samples per Frame")
        self.HWPframelabel.setBuddy(self.HWPframe)

        self.aHWPconv = QLineEdit('')
        self.bHWPconv = QLineEdit('')
        self.HWPconv = QLabel('HWP conversion factors')
        self.HWPconv.setBuddy(self.aHWPconv)

        self.HWPCheckBox.toggled.connect(self.HWPtypelabel.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPtype.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPfreqlabel.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPfreq.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPframelabel.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPframe.setVisible)
        self.HWPCheckBox.toggled.connect(self.HWPconv.setVisible)
        self.HWPCheckBox.toggled.connect(self.aHWPconv.setVisible)
        self.HWPCheckBox.toggled.connect(self.bHWPconv.setVisible)

        self.configuration_update()
        self.experiment.activated[str].connect(self.configuration_update)

        self.DirConvCheckBox.toggled.connect(self.detconv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.adetconv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bdetconv.setVisible)

        self.DirConvCheckBox.toggled.connect(self.coord1conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.acoord1conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bcoord1conv.setVisible)

        self.DirConvCheckBox.toggled.connect(self.coord2conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.acoord2conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bcoord2conv.setVisible)

        self.layout = QGridLayout()
        self.layout.addWidget(self.detfreqlabel, 1, 0)
        self.layout.addWidget(self.detfreq, 1, 1, 1, 3)
        self.layout.addWidget(self.acsfreqlabel, 2, 0)
        self.layout.addWidget(self.acsfreq, 2, 1, 1, 3)
        self.layout.addWidget(self.highpassfreqlabel, 3, 0)
        self.layout.addWidget(self.highpassfreq, 3, 1, 1, 3)
        self.layout.addWidget(self.detframelabel, 4, 0)
        self.layout.addWidget(self.detframe, 4, 1, 1, 3)
        self.layout.addWidget(self.acsframelabel, 5, 0)
        self.layout.addWidget(self.acsframe, 5, 1, 1, 3)
        self.layout.addWidget(self.dettypelabel, 6, 0)
        self.layout.addWidget(self.dettype, 6, 1, 1, 3)
        self.layout.addWidget(self.coord1typelabel, 7, 0)
        self.layout.addWidget(self.coord1type, 7, 1, 1, 3)
        self.layout.addWidget(self.coord2typelabel, 8, 0)
        self.layout.addWidget(self.coord2type, 8, 1, 1, 3)
        self.layout.addWidget(self.numberframelabel, 9, 0)
        self.layout.addWidget(self.startframe, 9, 2, 1, 1)
        self.layout.addWidget(self.endframe, 9, 3, 1, 1)

        self.layout.addWidget(self.DirConvCheckBox, 10, 0)
        self.layout.addWidget(self.detconv, 11, 1)
        self.layout.addWidget(self.adetconv, 11, 2)
        self.layout.addWidget(self.bdetconv, 11, 3)
        self.layout.addWidget(self.coord1conv, 12, 1)
        self.layout.addWidget(self.acoord1conv, 12, 2)
        self.layout.addWidget(self.bcoord1conv, 12, 3)
        self.layout.addWidget(self.coord2conv, 13, 1)
        self.layout.addWidget(self.acoord2conv, 13, 2)
        self.layout.addWidget(self.bcoord2conv, 13, 3)
        self.detconv.setVisible(True)
        self.adetconv.setVisible(True)
        self.bdetconv.setVisible(True)
        self.coord1conv.setVisible(True)
        self.acoord1conv.setVisible(True)
        self.bcoord1conv.setVisible(True)
        self.coord2conv.setVisible(True)
        self.acoord2conv.setVisible(True)
        self.bcoord2conv.setVisible(True)

        self.layout.addWidget(self.HWPCheckBox, 14, 0)
        self.layout.addWidget(self.HWPtypelabel, 15, 0)
        self.layout.addWidget(self.HWPtype, 15, 1, 1, 3)
        self.layout.addWidget(self.HWPfreqlabel, 16, 0)
        self.layout.addWidget(self.HWPfreq, 16, 1, 1, 3)
        self.layout.addWidget(self.HWPframelabel, 17, 0)
        self.layout.addWidget(self.HWPframe, 17, 1, 1, 3)
        self.layout.addWidget(self.HWPconv, 18, 1)
        self.layout.addWidget(self.aHWPconv, 18, 2)
        self.layout.addWidget(self.bHWPconv, 18, 3)

        if self.HWPCheckBox.isChecked() is True:
            self.HWPtypelabel.setVisible
            self.HWPtype.setVisible
            self.HWPfreqlabel.setVisible
            self.HWPfreq.setVisible
            self.HWPframelabel.setVisible
            self.HWPframe.setVisible
            self.HWPconv.setVisible
            self.aHWPconv.setVisible
            self.bHWPconv.setVisible
        else:
            self.HWPtypelabel.setVisible(False)
            self.HWPtype.setVisible(False)
            self.HWPfreqlabel.setVisible(False)
            self.HWPfreq.setVisible(False)
            self.HWPframelabel.setVisible(False)
            self.HWPframe.setVisible(False)
            self.HWPconv.setVisible(False)
            self.aHWPconv.setVisible(False)
            self.bHWPconv.setVisible(False)

        if self.DirConvCheckBox.isChecked():
            self.detconv.setVisible
            self.adetconv.setVisible
            self.bdetconv.setVisible
            self.coord1conv.setVisible
            self.acoord1conv.setVisible
            self.bcoord1conv.setVisible
            self.coord2conv.setVisible
            self.acoord2conv.setVisible
            self.bcoord2conv.setVisible
        else:
            self.detconv.setVisible(False)
            self.adetconv.setVisible(False)
            self.bdetconv.setVisible(False)
            self.coord1conv.setVisible(False)
            self.acoord1conv.setVisible(False)
            self.bcoord1conv.setVisible(False)
            self.coord2conv.setVisible(False)
            self.acoord2conv.setVisible(False)
            self.bcoord2conv.setVisible(False)

        #self.ExperimentGroup.setContentsMargins(5, 5, 5, 5)

        self.ExperimentGroup.setLayout(self.layout)

    def configuration_update(self):

        '''
        Function to update the experiment parameters based on some templates.
        It requires the coordinates system and the experiment name
        '''

        text = self.experiment.currentText()
        coord_text = self.coordchoice.currentText()

        self.configuration_value(text, coord_text)

    def configuration_value(self, text, coord_text):

        '''
        Function to read the experiment parameters from the template
        '''
        
        dir_path = os.getcwd()+'/config/'
        
        filepath = dir_path+text.lower()+'.cfg'
        model = configparser.ConfigParser()

        model.read(filepath)
        sections = model.sections()

        for section in sections:
            if section.lower() == 'experiment parameters':
                self.detfreq_config = float(model.get(section, 'DETFREQ').split('#')[0])
                det_dir_conv = model.get(section,'DET_DIR_CONV').split('#')[0].strip()
                self.detconv_config = np.array(det_dir_conv.split(',')).astype(float)
                self.detframe_config = float(model.get(section, 'DET_SAMP_FRAME').split('#')[0])
                self.dettype_config = model.get(section,'DET_FILE_TYPE').split('#')[0].strip()

            elif section.lower() == 'ra_dec parameters':
                if coord_text.lower() == 'ra and dec':
                    self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
                    coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
                    self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
                    coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
                    self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
                    self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
                    self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
                    self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()
                else:
                    pass

            elif section.lower() == 'az_el parameters':
                if coord_text.lower() == 'ra and dec' or coord_text.lower() == 'xy stage':
                    pass
                else:
                    self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
                    coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
                    self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
                    coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
                    self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
                    self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
                    self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
                    self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()

            elif section.lower() == 'xy_stage parameters':
                if coord_text.lower() == 'xy stage':
                    self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
                    coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
                    self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
                    coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
                    self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
                    self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
                    self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
                    self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()
                else:
                    pass

            elif section.lower() == 'hwp parameters':
                self.HWPfreq_config = float(model.get(section, 'HWPFREQ').split('#')[0])
                hwp_dir_conv = model.get(section,'HWP_DIR_CONV').split('#')[0].strip()
                self.HWPconv_config = np.array(hwp_dir_conv.split(',')).astype(float)
                self.HWPframe_config = float(model.get(section, 'HWP_SAMP_FRAME').split('#')[0])
                self.HWPtype_config = model.get(section,'HWP_FILE_TYPE').split('#')[0].strip()

            
        self.detfreq.setText(str(self.detfreq_config))
        self.acsfreq.setText(str(self.acsfreq_config))
        self.detframe.setText(str(self.detframe_config))
        self.acsframe.setText(str(self.acsframe_config))
        self.dettype.setText(str(self.dettype_config))
        self.coord1type.setText(str(self.coord1type_config))
        self.coord2type.setText(str(self.coord2type_config))
        self.adetconv.setText(str(self.detconv_config[0]))
        self.bdetconv.setText(str(self.detconv_config[1]))
        self.acoord1conv.setText(str(self.coord1conv_config[0]))
        self.bcoord1conv.setText(str(self.coord1conv_config[1]))
        self.acoord2conv.setText(str(self.coord2conv_config[0]))
        self.bcoord2conv.setText(str(self.coord2conv_config[1]))
        self.HWPfreq.setText(str(self.HWPfreq_config))
        self.HWPframe.setText(str(self.HWPframe_config))
        self.HWPtype.setText(str(self.HWPtype_config))
        self.aHWPconv.setText(str(self.HWPconv_config[0]))
        self.bHWPconv.setText(str(self.HWPconv_config[1]))
          
    def createOffsetGroup(self):

        '''
        Function to create the layout and the output of the offset group.
        Check the pointing.py for offset calculation
        '''


        self.OffsetGroup = QGroupBox("Detector Offset")

        self.CROSSELoffsetlabel = QLabel('Cross Elevation (deg)')
        self.ELxoffsetlabel = QLabel('Elevation (deg)')
        self.CROSSELoffset = QLineEdit('')
        self.ELxoffset = QLineEdit('')

        #self.coordchoice.activated[str].connect(self.updateOffsetLabel)

        #self.updateOffsetLabel()

        self.PointingOffsetCalculationCheckBox.toggled.connect(self.updateOffsetLabel)
        self.updateOffsetLabel()

        self.layout = QGridLayout()
        self.layout.addWidget(self.CROSSELoffsetlabel, 0, 0)
        self.layout.addWidget(self.CROSSELoffset, 0, 1)
        self.layout.addWidget(self.ELxoffsetlabel, 1, 0)
        self.layout.addWidget(self.ELxoffset, 1, 1)

        self.ELxoffset.setEnabled(False)
        self.CROSSELoffset.setEnabled(False)
        
        self.OffsetGroup.setLayout(self.layout)

    def updateOffsetLabel(self):

        '''
        Update the offset labels based on the coordinate system choice
        '''
        if self.PointingOffsetCalculationCheckBox.isChecked():
            self.OffsetGroup.setVisible(True)
        else:
            self.OffsetGroup.setVisible(False)

    def updateOffsetValue(self, coord1_ref=None, coord2_ref=None):

        '''
        Calculate and update the offset value based on the coordinate system choice
        '''

        if self.ctype.lower() == 'ra and dec':
            coord1_ref = copy.copy(float(self.crval1.text()))
            coord2_ref = copy.copy(float(self.crval2.text()))
        else:
            coord1_ref = coord1_ref
            coord2_ref = coord2_ref

        offset = pt.compute_offset(coord1_ref, coord2_ref, self.map_value, self.w[:,0], self.w[:,1],\
                                   self.proj, self.ctype, self.lstslice, self.latslice)

        
        xel_offset, el_offset = offset.value()

        self.CROSSELoffset.setText(str(xel_offset))
        self.CROSSELoffset.setStyleSheet("color: red;")
        self.ELxoffset.setText(str(el_offset))
        self.ELxoffset.setStyleSheet("color: red;")

    def load_func(self, offset = None, correction=False, LSTtype=None, LATtype=None,\
                  LSTconv=None, LATconv=None, lstlatfreq=None, lstlatsample = None, 
                  polynomialorder=int(5), despike=True, sigma=int(5), prominence=int(5)):


        '''
        Wrapper function to loaddata.py to read the DIRFILEs.
        If the paths are not correct a warning is generated. To reduce the time to 
        re-run the code everytime, a new DIRFILE is loaded a pickle object is created so it can be 
        loaded again when the plot button is pushed. The pickles object are deleted when
        the software is closed
        '''
        label_final = []
        coord_type = self.coordchoice.currentText()
        if coord_type == 'RA and DEC':
            self.coord1 = str('RA')
            self.coord2 = str('DEC')
        elif coord_type == 'AZ and EL':
            self.coord1 = str('AZ')
            self.coord2 = str('EL')
        elif coord_type == 'CROSS-EL and EL':
            self.coord1 = str('CROSS-EL')
            self.coord2 = str('EL')
        elif coord_type == 'XY Stage':
            self.coord1 = str('X')
            self.coord2 = str('Y')
        
        self.det_list = list(map(str.strip, self.detname.text().split(',')))
        
        for i in range(np.size(self.det_list)):
            if self.experiment.currentText().lower() == 'blast-tng':
                try:
                    list_conv = [['A', 'B'], ['D', 'E'], ['G', 'H'], ['K', 'I'], ['M', 'N']]
                    kid_num  = int(self.det_list[i])
                    det_I_string = 'kid'+list_conv[kid_num-1][0]+'_roachN'
                    print('DET', det_I_string)
                    os.stat(self.detpath.text()+'/'+det_I_string)
                except OSError:
                    label = self.detpathlabel.text()[:-1]+':'+self.det_list[i]
                    label_final.append(label)
            else:
                try:
                    os.stat(self.detpath.text()+'/'+self.det_list[i])
                except OSError:
                    label = self.detpathlabel.text()[:-1]+':'+self.det_list[i]
                    label_final.append(label)

        if self.experiment.currentText().lower() == 'blast-tng':    
            try:
                os.stat(self.coordpath.text())
            except OSError:
                label = self.coordpathlabel.text()
                label_final.append(label)
        elif self.experiment.currentText().lower() == 'blastpol':
            try:
                if self.coord1 == 'RA':
                    os.stat(self.coordpath.text()+'/'+self.coord1.lower())
                else:
                    os.stat(self.coordpath.text()+'/az')
            except OSError:
                label = self.coord1.lower()+' coordinate'
                label_final.append(label)
            try:
                os.stat(self.coordpath.text()+'/'+self.coord2.lower())
            except OSError:
                label = self.coord2.lower()+' coordinate'
                label_final.append(label)

        label_lst = []
        if correction:
            if self.PointingOffsetCalculationCheckBox.isChecked():
                try:
                    os.stat(os.getcwd()+'/xsc_'+self.pointingoffsetnumber.text()+'.txt')
                except OSError:
                    label = 'StarCamera'
                    label_final.append(label)
            if self.DettableCheckBox.isChecked():
                try:
                    os.stat(self.dettablepath.text()+'bolotable.tsv')
                except OSError:
                    label = 'BoloTable'
                    label_final.append(label)
        
        if (correction and self.coord1.lower() == 'ra') or self.telescopecoordinateCheckBox.isChecked():
            try:
                os.stat(self.coordpath.text()+'/'+'lst')
            except OSError:
                label = 'LST'
                label_final.append(label)
            try:
                os.stat(self.coordpath.text()+'/'+'lat')
            except OSError:
                label = 'LAT'
                label_final.append(label)

            if LSTtype is None:
                label_lst = 'Write LST and LAT Parameters from the menubar'

        if np.size(label_final)+np.size(label_lst) != 0:
            if np.size(label_final) != 0:
                self.warningbox = QMessageBox()
                self.warningbox.setIcon(QMessageBox.Warning)
                self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                self.warningbox.setWindowTitle('Warning')

                msg = 'Incorrect Path(s): \n'
                for i in range(len(label_final)): 
                    msg += (str(label_final[i])) +'\n'
                
                self.warningbox.setText(msg)        
            
                self.warningbox.exec_()
            
            if np.size(label_lst) !=0:
                self.lstwarningbox = QMessageBox()
                self.lstwarningbox.setIcon(QMessageBox.Warning)
                self.lstwarningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                self.lstwarningbox.setWindowTitle('Warning')
                
                self.lstwarningbox.setText(label_lst)        
            
                self.lstwarningbox.exec_()

        else:
            
            os.makedirs(os.path.dirname('pickles_object/'), exist_ok=True)

            det_path_pickle = []
            for i in range(np.size(self.det_list)):
                det_path_pickle.append('pickles_object/'+self.det_list[i])
            coord1_path_pickle = 'pickles_object/'+self.coord1
            coord2_path_pickle = 'pickles_object/'+self.coord2
            hwp_path_pickle = 'pickles_object/hwp'
            lat_path_pickle = 'pickles_object/lat'
            lst_path_pickle = 'pickles_object/lst'

            try:
                for i in range(np.size(self.det_list)):
                    if i == 0:               
                        self.det_data = pickle.load(open(det_path_pickle[0], 'rb'))
                    else:
                        self.det_data = np.vstack((self.det_data, pickle.load(open(det_path_pickle[i], 'rb')))) 
                self.coord1_data = pickle.load(open(coord1_path_pickle, 'rb'))
                self.coord2_data = pickle.load(open(coord2_path_pickle, 'rb'))


                if self.HWPCheckBox.isChecked():
                    self.hwp_data = pickle.load(open(hwp_path_pickle, 'rb'))
                else:
                    self.hwp_data = None
                
                if (correction and self.coord1.lower() == 'ra') or self.telescopecoordinateCheckBox.isChecked():
                    self.lst_data = pickle.load(open(lst_path_pickle, 'rb'))
                    self.lat_data = pickle.load(open(lat_path_pickle, 'rb'))
                else:
                    self.lst_data = None
                    self.lat_data = None

            except FileNotFoundError:

                if (correction and self.coord1.lower() == 'ra') or self.telescopecoordinateCheckBox.isChecked():
                    lat_file_type = LSTtype
                    lst_file_type = LATtype
                else: 
                    lat_file_type = None
                    lst_file_type = None


                if self.HWPCheckBox.isChecked():
                    hwptype = self.HWPtype.text()
                else:
                    hwptype=None

                dataload = ld.data_value(self.detpath.text(), self.det_list, self.coordpath.text(), \
                                         self.coord1, self.coord2, self.dettype.text(), \
                                         self.coord1type.text(), self.coord2type.text(), \
                                         self.experiment.currentText(), lst_file_type, lat_file_type, hwptype, 
                                         self.startframe.text(), self.endframe.text())
                

                

                if (correction and self.coord1.lower() == 'ra') or self.telescopecoordinateCheckBox.isChecked():
                    (self.det_data, self.coord1_data, self.coord2_data, self.hwp_data, \
                     self.lst_data, self.lat_data) = dataload.values()

                    pickle.dump(self.lst_data, open(lst_path_pickle, 'wb'))
                    pickle.dump(self.lat_data, open(lat_path_pickle, 'wb'))
                else:
                    self.det_data, self.coord1_data, self.coord2_data, self.hwp_data = dataload.values()
                    self.lst_data = None
                    self.lat_data = None

                for i in range(np.size(self.det_list)):
                    if np.size(np.shape(self.det_data)) > 1:

                        pickle.dump(self.det_data[i,:], open(det_path_pickle[i], 'wb'))
                    else:
                        pickle.dump(self.det_data, open(det_path_pickle[i], 'wb'))

                pickle.dump(self.coord1_data, open(coord1_path_pickle, 'wb'))
                pickle.dump(self.coord2_data, open(coord2_path_pickle, 'wb'))

                if self.HWPCheckBox.isChecked():
                    pickle.dump(self.hwp_data, open(hwp_path_pickle, 'wb'))
                
                del dataload
                gc.collect()

            if self.experiment.currentText().lower() == 'blast-tng':
                if coord_type == 'XY Stage':
                    xystage = True
                else:
                    xystage = False

                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  self.lst_data, self.lat_data, lstlatfreq, \
                                                  lstlatsample, offset=offset, \
                                                  roach_number = self.roachnumber.text(), \
                                                  roach_pps_path = self.detpath.text(), xystage=xystage)

            elif self.experiment.currentText().lower() == 'blastpol':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  self.lst_data, self.lat_data, lstlatfreq, \
                                                  lstlatsample, offset, hwp_data=self.hwp_data, hwp_fs=self.HWPfreq.text(),\
                                                  hwp_sample_frame=self.HWPframe.text())

            if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked() or \
               self.telescopecoordinateCheckBox.isChecked():
                (self.timemap, self.detslice, self.coord1slice, \
                 self.coord2slice, self.hwpslice, self.lstslice, self.latslice) = zoomsyncdata.sync_data()
            else:
                (self.timemap, self.detslice, self.coord1slice, \
                 self.coord2slice, self.hwpslice) = zoomsyncdata.sync_data()
                self.lstslice = None
                self.latslice = None

            del self.det_data
            gc.collect()
            if self.DirConvCheckBox.isChecked():
                self.dirfile_conversion(correction = correction, LSTconv = LSTconv, \
                                        LATconv = LATconv)
            else:
                if self.HWPCheckBox.isChecked():
                    if self.experiment.currentText().lower() == 'blastpol':
                        self.hwpslice = (self.hwpslice-0.451)*(-360.)

            if self.DettableCheckBox.isChecked():
                dettable = ld.det_table(self.det_list, self.experiment.currentText(), self.dettablepath.text())
                self.det_off, self.noise_det, self.grid_angle, self.pol_angle_offset, self.resp = dettable.loadtable()
            else:
                self.det_off = np.zeros((np.size(self.det_list),2))
                self.noise_det = np.ones(np.size(self.det_list))
                self.grid_angle = np.zeros(np.size(self.det_list))
                self.pol_angle_offset = np.zeros(np.size(self.det_list))
                self.resp = np.ones(np.size(self.det_list))
                        
            if self.coord1.lower() == 'cross-el':
                self.coord1slice = self.coord1slice*np.cos(np.radians(self.coord2slice))

            if correction is True:
                if self.PointingOffsetCalculationCheckBox.isChecked():               
                    xsc_file = ld.xsc_offset(self.pointingoffsetnumber.text(), self.startframe.text(), self.endframe.text())
                    xsc_offset = xsc_file.read_file()
                else:
                    xsc_offset = np.zeros(2)

                corr = pt.apply_offset(self.coord1slice, self.coord2slice, coord_type,\
                                       xsc_offset, det_offset = self.det_off, lst = self.lstslice, \
                                       lat = self.latslice)

                self.coord1slice, self.coord2slice = corr.correction()
            else:
                if self.coord1.lower() == 'ra':
                    self.coord1slice = self.coord1slice*15. #Conversion between hours to degree

            if self.telescopecoordinateCheckBox.isChecked() or self.ICheckBox.isChecked() is False:
                
                self.parallactic = np.zeros_like(self.coord1slice)
                if np.size(np.shape(self.detslice)) == 1:
                    tel = pt.utils(self.coord1slice/15., self.coord2slice, \
                                   self.lstslice, self.latslice)
                    self.parallactic = tel.parallactic_angle()
                else:
                    if np.size(np.shape(self.coord1slice)) == 1:
                        tel = pt.utils(self.coord1slice/15., self.coord2slice, \
                                       self.lstslice, self.latslice)
                        self.parallactic = tel.parallactic_angle()
                    else:
                        for i in range(np.size(np.shape(self.detslice))):
                            tel = pt.utils(self.coord1slice[i]/15., self.coord2slice[i], \
                                           self.lstslice, self.latslice)
                            self.parallactic[i,:] = tel.parallactic_angle()
            else:
                if np.size(np.shape(self.detslice)) == 1:
                    self.parallactic = 0.
                else:
                    if np.size(np.shape(self.coord1slice)) == 1:
                        self.parallactic = 0.
                    else:
                        self.parallactic = np.zeros_like(self.detslice)
            del self.coord1_data
            del self.coord2_data 
            del zoomsyncdata
            gc.collect()

            self.clean_func(polynomialorder, despike, sigma, prominence)

    def clean_func(self, polynomialorder, despike, sigma, prominence):

        '''
        Function to compute the cleaned detector TOD
        '''

        det_tod = tod.data_cleaned(self.detslice, self.detfreq.text(), self.highpassfreq.text(), self.det_list,
                                   polynomialorder, despike, sigma, prominence)
        self.cleaned_data = det_tod.data_clean()

        if np.size(self.resp) > 1:
            if self.experiment.currentText().lower() == 'blast-tng':
                self.cleaned_data = np.multiply(self.cleaned_data, np.reshape(1/self.resp, (np.size(1/self.resp), 1)))
            else:
                self.cleaned_data = np.multiply(self.cleaned_data, np.reshape(self.resp, (np.size(self.resp), 1)))
        else:
            if self.experiment.currentText().lower() == 'blast-tng':
                self.cleaned_data /= self.resp
            else:
                self.cleaned_data *= self.resp

    def dirfile_conversion(self, correction=False, LSTconv=None, LATconv=None):

        '''
        Function to convert the DIRFILE data.
        '''

        det_conv = ld.convert_dirfile(self.detslice, float(self.adetconv.text()), \
                                      float(self.bdetconv.text()))
        coord1_conv = ld.convert_dirfile(self.coord1slice, float(self.acoord1conv.text()), \
                                         float(self.bcoord1conv.text()))
        coord2_conv = ld.convert_dirfile(self.coord2slice, float(self.acoord2conv.text()), \
                                         float(self.bcoord2conv.text()))

        det_conv.conversion()
        coord1_conv.conversion()
        coord2_conv.conversion()

        self.detslice = det_conv.data
        self.coord1slice = coord1_conv.data
        self.coord2slice = coord2_conv.data

        if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked() \
           or self.telescopecoordinateCheckBox.isChecked():
            lst_conv = ld.convert_dirfile(self.lstslice, float(LSTconv[0]), \
                                          float(LSTconv[1]))
            lat_conv = ld.convert_dirfile(self.latslice, float(LATconv[0]), \
                                          float(LATconv[1]))

            lst_conv.conversion()
            lat_conv.conversion()

            self.lstslice = lst_conv.data
            self.latslice = lat_conv.data

        if self.HWPCheckBox.isChecked():
            hwp_conv = ld.convert_dirfile(self.hwpslice, float(self.aHWPconv.text()), \
                                          float(self.bHWPconv.text()))

            hwp_conv.conversion()
            if self.experiment.currentText().lower() == 'blastpol':
                self.hwpslice = (hwp_conv.data-0.451)*(-360.)
            
    def mapvalues(self, data):

        '''
        Function to compute the maps
        '''

        self.ctype = self.coordchoice.currentText()

        self.crpix = np.array([int(float(self.crpix1.text())),\
                               int(float(self.crpix2.text()))])
        self.cdelt = np.array([float(self.cdelt1.text()),\
                               float(self.cdelt2.text())])
        # if self.crval is None:
        self.crval = np.array([float(self.crval1.text()),\
                               float(self.crval2.text())])
        self.pixnum = np.array([float(self.pixnum1.text()),\
                               float(self.pixnum2.text())])

        if self.convchoice.currentText().lower() == 'gaussian':
            self.convolution = True
            self.std = self.GaussianSTD.text()
        else:
            self.convolution = False
            self.std = 0

        #Compute final polarization angle

        if np.size(self.det_list) == 1:
            self.pol_angle = np.radians(self.parallactic+2*self.hwpslice+(self.grid_angle-2*self.pol_angle_offset))
            if np.size(np.shape(self.coord1slice)) != 1:
                self.pol_angle = np.reshape(self.pol_angle, np.size(self.pol_angle))
        else:
            self.pol_angle = np.zeros_like(data)
            for i in range(np.size(self.det_list)):
                self.pol_angle[i,:] = np.radians(2*self.hwpslice+(self.grid_angle[i]-2*self.pol_angle_offset[i]))
                if np.size(np.shape(self.coord1slice)) == 1:
                    self.pol_angle[i, :] += np.radians(self.parallactic)
                else:
                    self.pol_angle[i, :] += np.radians(self.parallactic[i,:])


        self.maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
                            data, self.coord1slice, self.coord2slice, \
                            self.convolution, self.std, self.ICheckBox.isChecked(), \
                            pol_angle=self.pol_angle, noise=self.noise_det, \
                            telcoord = self.telescopecoordinateCheckBox.isChecked(), \
                            parang=self.parallactic)

        self.maps.wcs_proj()

        self.proj = self.maps.proj
        self.w = self.maps.w

        self.map_value = self.maps.map2d()

class TODTab(QWidget):

    '''
    Layout Class for the TOD tab
    '''

    def __init__(self, parent=None):

        super(QWidget, self).__init__(parent)

        self.c = ParamMapTab()

        self.createTODplot()
        self.createTODcleanedplot()

        self.detcombolist = QComboBox()

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.detcombolist, 0, 0)
        mainlayout.addWidget(self.TODplot, 1, 0)
        mainlayout.addWidget(self.TODcleanedplot, 1, 1)
        
        self.setLayout(mainlayout)

    def createTODplot(self, data = None):

        '''
        Function to create the TOD empty plot
        '''

        self.TODplot = QGroupBox("Detector TOD")
        TODlayout = QGridLayout()

        self.matplotlibWidget_TOD = MatplotlibWidget(self)
        self.axis_TOD = self.matplotlibWidget_TOD.figure.add_subplot(111)
        self.axis_TOD.set_axis_off()
        TODlayout.addWidget(self.matplotlibWidget_TOD)        

        self.TODplot.setLayout(TODlayout)

    def draw_TOD(self, data = None):

        '''
        Function to draw the TOD when the plot button is pushed.
        The plotted TOD is the one between the frame of interest
        '''
        
        self.axis_TOD.set_axis_on()
        self.axis_TOD.clear()
        try:
            self.axis_TOD.plot(data)
        except AttributeError:
            pass
        self.axis_TOD.set_title('detTOD')
        self.matplotlibWidget_TOD.canvas.draw()

    def createTODcleanedplot(self, data = None):

        '''
        Same of createTODPlot but for the cleanedTOD
        '''

        self.TODcleanedplot = QGroupBox("Detector Cleaned TOD")
        self.layout = QVBoxLayout()

        self.matplotlibWidget_cleaned_TOD = MatplotlibWidget(self)
        self.axis_cleaned_TOD = self.matplotlibWidget_cleaned_TOD.figure.add_subplot(111)
        self.axis_cleaned_TOD.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_cleaned_TOD)

        self.TODcleanedplot.setLayout(self.layout)

    def draw_cleaned_TOD(self, data = None):

        '''
        Same of draw_TOD but for the cleaned TOD
        '''
        
        self.axis_cleaned_TOD.set_axis_on()
        self.axis_cleaned_TOD.clear()
        try:           
            self.axis_cleaned_TOD.plot(data)
        except AttributeError or NameError or TypeError:
            pass
        self.axis_cleaned_TOD.set_title('Cleaned Data')
        self.matplotlibWidget_cleaned_TOD.canvas.draw()

    def detlist_selection(self, detlist):
        self.detcombolist.clear()
        if np.size(detlist) == 1:
            self.detcombolist.addItem(str(detlist))
        else:
            for i in range(np.size(detlist)):
                self.detcombolist.addItem(str(detlist[i]))

class BeamTab(ParamMapTab):

    '''
    Layout for the tab used to show the calculated beams
    '''

    def __init__(self, parent=None, checkbox=None, ctype=None):

        super(QWidget, self).__init__(parent)

        self.beammaps = MapPlotsGroup(checkbox=checkbox, data=None)

        self.table = QTableWidget()

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.beammaps, 0, 0)
        mainlayout.addWidget(self.table, 1, 0)
        
        self.setLayout(mainlayout)

    def configureTable(self, table, rows):
        table.setColumnCount(6)
        table.setRowCount(rows)
        table.setHorizontalHeaderItem(0, QTableWidgetItem("Amplitude"))
        table.setHorizontalHeaderItem(1, QTableWidgetItem("X0 (in pixel)"))
        table.setHorizontalHeaderItem(2, QTableWidgetItem("Y0 (in pixel)"))
        table.setHorizontalHeaderItem(3, QTableWidgetItem("SigmaX (in pixel)"))
        table.setHorizontalHeaderItem(4, QTableWidgetItem("SigmaY (in pixel)"))
        table.setHorizontalHeaderItem(5, QTableWidgetItem("Theta"))

    def updateTable(self, gauss_param):
        try:
            self.configureTable(self.table, int(np.size(gauss_param)/6))
            self.updateParamValues(gauss_param)
        except ValueError:
            pass

    def updateParamValues(self, gauss_param):

        rows_number = self.table.rowCount()
        column_number = self.table.columnCount()
        count =0
        for i in range(rows_number):
            for j in range(column_number):
                print('OK')
                self.table.setItem(i,j, QTableWidgetItem(str(format(gauss_param[count],'.3f'))))
                count +=1

class MapPlotsGroup(QWidget):

    '''
    Generic layout to create a tabbed plot layout for maps
    in case only I is requested or also polarization maps
    are requested as output.
    This class is used for plotting both the maps and the beams 
    '''

    def __init__(self, data, checkbox, parent=None):

        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        
        self.data = data
        self.checkbox = checkbox
        self.cutout = None

        self.tabvisible()

        self.tabs.addTab(self.tab1,"I Map")
        self.ImapTab()
        self.QmapTab()
        self.UmapTab()

        self.checkbox.toggled.connect(self.tabvisible)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def tabvisible(self):

        '''
        Function to update the visibility of the polarization maps.
        '''

        if self.checkbox.isChecked():
            self.Qsave = self.tabs.widget(1)
            self.tabs.removeTab(1)
            self.Usave = self.tabs.widget(1)
            self.tabs.removeTab(1)
        else:
            try:
                self.tabs.insertTab(1, self.Qsave)
                self.tabs.insertTab(2, self.Usave)
            except:
                self.tabs.addTab(self.tab2,"Q Map")
                self.tabs.addTab(self.tab3,"U Map")

    def ImapTab(self, button=None):

        '''
        Create an empty plot for I map (or beam if the class is used in the beam tab)
        '''

        mainlayout = QGridLayout()

        self.matplotlibWidget_Imap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Imap)

        self.tab1.setLayout(mainlayout)

    def QmapTab(self):

        '''
        Same of ImapTab but the Q Stokes parameter
        '''

        mainlayout = QGridLayout()

        self.matplotlibWidget_Qmap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Qmap)

        self.tab2.setLayout(mainlayout)

    def UmapTab(self):
        
        '''
        Same of ImapTab but the U Stokes parameter
        '''

        self.UmapGroup = QGroupBox("Detector Offset")
        mainlayout = QGridLayout()

        self.matplotlibWidget_Umap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Umap)

        self.tab3.setLayout(mainlayout)

    def updateTab(self, data, coord1, coord2, crval, ctype, pixnum, telcoord=False, cdelt=None, crpix=None, \
                  projection=None, xystage=False):

        '''
        Function to updates the I, Q and U plots when the 
        button plot is pushed
        '''
        
        if np.size(np.shape(data)) > 2:
            idx_list = ['I', 'Q', 'U']

            for i in range(len(idx_list)):
                self.map2d(data=data[i], coord1=coord1, coord2=coord2, crval=crval, ctype=ctype, pixnum=pixnum, idx=idx_list[i],\
                           telcoord=telcoord, cdelt=cdelt, crpix=crpix, projection=projection, xystage=xystage)
        else:
            self.map2d(data=data, coord1=coord1, coord2=coord2, crval=crval, ctype= ctype, pixnum=pixnum, telcoord=telcoord, cdelt=cdelt, crpix=crpix, \
                       projection=projection,xystage=xystage)

    def map2d(self, data=None, coord1=None, coord2=None, crval=None, ctype=None, pixnum=None, idx='I', telcoord=False, cdelt=None, \
              crpix=None, projection=None, xystage=False):

        '''
        Function to generate the map plots (I,Q and U) 
        when the plot button is pushed
        '''
        
        intervals = 3   

        if telcoord is False:        
            if xystage is False:
                position = SkyCoord(crval[0], crval[1], unit='deg', frame='icrs')

                size = (pixnum[1], pixnum[0])     # pixels

                cutout = Cutout2D(data, position, size, wcs=projection)
                proj = cutout.wcs
                print('PROJ', proj)
                self.mapdata = cutout.data
            else:
                masked = np.ma.array(data, mask=(np.abs(data)<1))
                self.mapdata = masked
                proj = 'rectilinear'

        else:
            idx_xmin = crval[0]-cdelt*pixnum[0]/2   
            idx_xmax = crval[0]+cdelt*pixnum[0]/2
            idx_ymin = crval[1]-cdelt*pixnum[1]/2
            idx_ymax = crval[1]+cdelt*pixnum[1]/2

            proj = None

            idx_xmin = np.amax(np.array([np.ceil(crpix[0]-1-pixnum[0]/2), 0.], dtype=int))
            idx_xmax = np.amin(np.array([np.ceil(crpix[0]-1+pixnum[0]/2), np.shape(data)[1]], dtype=int))

            if np.abs(idx_xmax-idx_xmin) != pixnum[0]:
                if idx_xmin != 0 and idx_xmax == np.shape(data)[1]:
                    idx_xmin = np.amax(np.array([0., np.shape(data)[1]-pixnum[0]], dtype=int))
                if idx_xmin == 0 and idx_xmax != np.shape(data)[1]:
                    idx_xmax = np.amin(np.array([pixnum[0], np.shape(data)[1]], dtype=int))

            idx_ymin = np.amax(np.array([np.ceil(crpix[1]-1-pixnum[1]/2), 0.], dtype=int))
            idx_ymax = np.amin(np.array([np.ceil(crpix[1]-1+pixnum[1]/2), np.shape(data)[0]], dtype=int))

            if np.abs(idx_ymax-idx_ymin) != pixnum[1]:
                if idx_ymin != 0 and idx_ymax == np.shape(data)[0]:
                    idx_ymin = np.amax(np.array([0., np.shape(data)[0]-pixnum[1]], dtype=int))
                if idx_ymin == 0 and idx_ymax != np.shape(data)[0]:
                    idx_ymax = np.amin(np.array([pixnum[1], np.shape(data)[0]], dtype=int))

            self.mapdata = data[idx_ymin:idx_ymax, idx_xmin:idx_xmax]
            crpix[0] -= idx_xmin
            crpix[1] -= idx_ymin

            w = wcs.WCS(naxis=2)
            w.wcs.crpix = crpix
            w.wcs.cdelt = cdelt
            w.wcs.crval = crval
            w.wcs.ctype = ["TLON-TAN", "TLAT-TAN"]
            proj = w

        levels = np.linspace(0.5, 1, intervals)*np.amax(self.mapdata)
        print('CTYPE', ctype)
        if idx == 'I':
            self.matplotlibWidget_Imap.figure.clear()
            fig = self.matplotlibWidget_Imap.figure
            if ctype == 'XY Stage':
                self.axis_Imap = fig.add_subplot(111, projection='rectilinear')
            else:
                self.axis_Imap = fig.add_subplot(111, projection=proj)
            axis = self.axis_Imap
        elif idx == 'Q':
            self.matplotlibWidget_Qmap.figure.clear()
            fig = self.matplotlibWidget_Qmap.figure
            self.axis_Qmap = fig.add_subplot(111, projection=proj)
            axis = self.axis_Qmap
        elif idx == 'U':
            self.matplotlibWidget_Umap.figure.clear()
            fig = self.matplotlibWidget_Umap.figure
            self.axis_Umap = fig.add_subplot(111, projection=proj)

            axis = self.axis_Umap

        if telcoord is False:
            if ctype == 'RA and DEC':
                ra = axis.coords[0]
                dec = axis.coords[1]
                ra.set_axislabel('RA (deg)')
                dec.set_axislabel('Dec (deg)')
                dec.set_major_formatter('d.ddd')
                ra.set_major_formatter('d.ddd')
            
            elif ctype == 'AZ and EL':
                az = axis.coords[0]
                el = axis.coords[1]
                az.set_axislabel('AZ (deg)')
                el.set_axislabel('EL (deg)')
                az.set_major_formatter('d.ddd')
                el.set_major_formatter('d.ddd')
            
            elif ctype == 'CROSS-EL and EL':
                xel = axis.coords[0]
                el = axis.coords[1]
                xel.set_axislabel('xEL (deg)')
                el.set_axislabel('EL (deg)')
                xel.set_major_formatter('d.ddd')
                el.set_major_formatter('d.ddd')

            elif ctype == 'XY Stage':
                axis.set_title('XY Stage')
                axis.set_xlabel('X')
                axis.set_ylabel('Y')

        else:
            ra_tel = axis.coords[0]
            dec_tel = axis.coords[1]
            ra_tel.set_axislabel('YAW (deg)')
            dec_tel.set_axislabel('PITCH (deg)')
            ra_tel.set_major_formatter('d.ddd')
            dec_tel.set_major_formatter('d.ddd')

        
        img = axis.images
        if np.size(img) > 0:
            cb = img[-1].colorbar
            cb.remove()
        axis.set_axis_on()

        if telcoord is False:
            im = axis.imshow(self.mapdata, origin='lower', cmap=plt.cm.viridis)
            axis.contour(self.mapdata, levels=levels, colors='white', alpha=0.5)
        else:
            im = axis.imshow(self.mapdata, origin='lower', cmap=plt.cm.viridis)
            axis.contour(self.mapdata, levels=levels, colors='white', alpha=0.5)
        plt.colorbar(im, ax=axis)
        #axis.set_axis_on()
        

        if idx == 'I':
            fig.canvas.draw()       
        elif idx == 'Q':
            fig.canvas.draw()
        elif idx == 'U':
            fig.canvas.draw()

class MatplotlibWidget(QWidget):

    '''
    Class to generate an empty matplotlib.pyplot object
    '''

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layoutVertical = QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)
        self.layoutVertical.addWidget(self.toolbar)



