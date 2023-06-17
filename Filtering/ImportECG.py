import time
from pySerialTransfer import pySerialTransfer as txfer
import csv
from UPDATE import DynamicUpdate

def import_all(COMport = '/dev/ttyACM0'):
    print("Importing ECG, GSR and respiratory data \n")
    d = DynamicUpdate(700)
    d.on_launch()
    try:
        link = txfer.SerialTransfer(COMport, baud=57600)

        open_link = link.open()
        print(f"Link is open: {open_link}")

        header = ['TimeStamp', 'GSR Data', 'ECG Data', 'EMG Data','Acc1 X', 'Acc1 Y', 'Acc1 z', 'Acc2 X', 'Acc2 Y', 'Acc2 z', 'Temperature']
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        output_file = open('ECGdata.csv', 'w')
        writer = csv.writer(output_file)
        writer.writerow(header)

        time.sleep(3)
        print(f"link status: {link.status}")       
        
        while True:
            if link.available():
                recSize = 0
                # Import time stamp of the data sample
                data[0] = link.rx_obj(obj_type='L', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['L']

                # Import ECG data from serial connection
                data[1] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

                # Import GSR data from serial connection
                data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
                
                #import EMG data from serial connection
                data[3] = link.rx_obj(obj_type='H', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
                
                #import ACC1 x data from serial connection
                data[4] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC1 y data from serial connection
                data[5] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC1 z data from serial connection
                data[6] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 x data from serial connection
                data[7] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 y data from serial connection
                data[8] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import ACC2 z data from serial connection
                data[9] = link.rx_obj(obj_type='h', start_pos=recSize)
                recSize += txfer.STRUCT_FORMAT_LENGTHS['h']
                
                #import TEMP data from serial connection
                data[10] = link.rx_obj(obj_type='H', start_pos=recSize)/100
                recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
                
                d.go(data[0],data[1])
                
                writer.writerow(data)

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

