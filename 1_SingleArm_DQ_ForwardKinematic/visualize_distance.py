'''
Plot figure visualizing distance to target curves of the MMC network.
Version from October 2016

@author: mschilling
'''
import matplotlib.pylab as py
import matplotlib.pyplot as plt
import numpy

class DataVisualizerPlot:
    # Drawing function only
    def __init__(self, leg_s = [1.0, 1.0 , 1.0]):
        # Initialisation of the segment lengths (as argument list given to the object)
        self.leg_segm = leg_s
        
    def draw_distance_curve(self, mean_distance_data=0, std_dev_low=0, std_dev_up=0 ):
        """
            The distance to target over time is drawn.
            Data has to be acquired by calling 
            store_endpoint_data - which
            is automatically called for the dynamic version.
        """
        py.ion()
        py.rcParams['figure.figsize'] = 6,4
        fig = plt.figure()
        py.plot([0,25], [1,0], linewidth=2.0, color='gray', alpha=0.7,marker='')
        py.plot(mean_distance_data, color='0.1', linewidth=2.0)
        py.hold(True)
        py.plot(std_dev_low,'--',color='0.5', linewidth=1.0)
        py.plot(std_dev_up,'--', color='0.5', linewidth=1.0)
        py.xlabel('Iteration Steps')
        py.ylabel('Distance (normalized)')
        py.title('Distance to Target (End point)')
        py.grid(True) 
        py.savefig("Results/Fig_FWKinematic_distance_time.pdf")
        
    def draw_distance_fingertips_curve(self, mean_distance_data=0, std_dev_low=0, std_dev_up=0, name="distanceBoth" ):
        """
            Plot mean distance over time between fingertips of two arms.
        """
        py.ion()
        py.rcParams['figure.figsize'] = 6,4
        fig = plt.figure()
        #py.plot([0,25], [1,0], linewidth=2.0, color='gray', alpha=0.7,marker='')
        py.plot(mean_distance_data, color='0.1', linewidth=2.0)
        py.hold(True)
        py.plot(std_dev_low,'--',color='0.5', linewidth=1.0)
        py.plot(std_dev_up,'--', color='0.5', linewidth=1.0)
        py.xlabel('Iteration Steps')
        py.ylabel('Distance (in units)')
        #py.title('Distance between right and left manipulator end points')
        py.grid(True) 
        py.savefig("Results/" + name)