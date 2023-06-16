import time
import numpy as np
from pySerialTransfer import pySerialTransfer as txfer
from multiprocessing import Process, Manager
from UPDATE import DynamicUpdate
import csv

def import_all(plotQueue, MLQueue):
    print("Importing ECG, GSR and respiratory data \n")
    try:
        link = txfer.SerialTransfer(port='COM3', baud=230400)

        open_link = link.open()
        print(f"Link is open: {open_link}")

        header = ['TimeStamp', 'ECG', 'EMG', 'Temp', 'EDA','ACC1 X', 'ACC1 Y', 'ACC1 Z', 'ACC2 X', 'ACC2 Y', 'ACC2 Z']
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        output_file = open('ECGdata.csv', 'w')
        writer = csv.writer(output_file)
        writer.writerow(header)
        
        MLarray = np.zeros(1,len(header))

        time.sleep(3)
        print(f"link status: {link.status}")       
        
        while True:
            packetSize = link.available()
            
            if packetSize > 0:
                recSize = 0
                # Import time stamp of the data sample
                data[0] = link.rx_obj(obj_type='L', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['L']

                # Import ECG data from serial connection
                data[1] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

                #import EMG data from serial connection
                data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
                
                #import TEMP data from serial connection
                data[3] = link.rx_obj(obj_type='H', start_pos=recSize)/100
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

                writer.writerow(data)
                
            if packetSize == 24:  
                # Import GSR data from serial connection
                data[4] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
                
                #import ACC1 x data from serial connection
                data[5] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC1 y data from serial connection
                data[6] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC1 z data from serial connection
                data[7] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 x data from serial connection
                data[8] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 y data from serial connection
                data[9] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 z data from serial connection
                data[10] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                plotQueue.put([data[0], data[1]])
                
            #add another row each time data is received.    
            MLarray = np.concatenate((MLarray, np.array(data)[np.newaxis, :]), axis=0)
                
            sendinterval = 10e6 #10s in microseconds
            if MLarray[1,0] - MLarray[-1,0] > sendinterval :
                MLQueue.put(MLarray)
                MLQueue = np.zeros((1, len(header)))

            elif link.status < 0:
                if link.status == txfer.CRC_ERROR:
                    print('ERROR: CRC_ERROR')
                elif link.status == txfer.PAYLOAD_ERROR:
                    print('ERROR: PAYLOAD_ERROR')
                elif link.status == txfer.STOP_BYTE_ERROR:
                    print('ERROR: STOP_BYTE_ERROR')
                else:
                    print('ERROR: {}'.format(link.status))                   

    except KeyboardInterrupt:
        link.close()
        output_file.close()
        print("Serial connection & output_file closed. \n Don't forget to rename ECGdata to prevent overwriting!")

def livePlotter(queue):
    d = DynamicUpdate(700)
    d.on_launch()
    while(1):
        if(not queue.empty()):
            data = queue.get()
            d.go(data[0]/1000000,data[1])
    
#main function, which starts all three parallel processes
if __name__ == "__main__":
   gender = input("Please input your gender (M/F) to ensure stress detection:\n") 
   if gender != 'M' or gender != 'F':
       print("Character not recognised, program will exit shortly.")
       time.sleep(3)
       exit()
    
   plotQueue = Manager().Queue() #queue which is responsible for sharing the data between the serial connection and the live plot
   MLQueue = Manager().Queue() #queue which is responsible for sharing data between the serial connection and machine learning program
   dataAcProcess = Process(target=import_all, args=(plotQueue, MLQueue, ))
   dataAcProcess.start()
   
   plotterProcess = Process(target=livePlotter, args=(plotQueue,))   
   plotterProcess.start()
   
   MLProcess = Process(target=EnterMLFunctionNameHere, args=(MLQueue, gender, ))
   
   
   
   
   
   
   
   
   
   
   
   
   
   