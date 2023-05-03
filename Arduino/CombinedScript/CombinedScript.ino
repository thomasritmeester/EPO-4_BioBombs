#include "SerialTransfer.h"

SerialTransfer myTransfer;

bool calibrated = false;
int Fs = 100; //Hz
unsigned long prevMillis;

const int ECG=A0;
const int GSR=A1;
const int EMG=A2;
uint16_t ecgValue=0;
uint16_t gsrValue=0;
uint16_t emgValue=0;
uint16_t gsr_average=0;
uint16_t ecg_average=0;
uint16_t emg_average=0;

struct STRUCT {
unsigned long timestamp;
uint16_t  ECG_data;
uint16_t GSR_data;
uint16_t EMG_data;
} dataPacket;

void setup()
{
  Serial.begin(57600);
  myTransfer.begin(Serial);
}

void loop()
{
  if(!calibrated && myTransfer.available()){
    for(uint16_t i=0; i < myTransfer.bytesRead; i++){
      if(myTransfer.packet.rxBuff[i] == 'k'){
        //insert calibration function here
        calibrated = true;
      }
    }
  }

  if(calibrated){
    takeMeasurements();

    // hold the program until 1/Fs has passed since the previous data packet was sent
    while(1){
      if((millis() - prevMillis) >= (1000/Fs)){
        prevMillis = millis();
        break;
      }
    }
    //send the actual data  
    sendData();
  }
}


void takeMeasurements(){
  //take 10 measurements and average them
  int ecg_sum=0, gsr_sum=0, emg_sum=0;
  for(int i=0;i<10;i++)           
    {
      gsrValue=analogRead(GSR);
      gsr_sum += gsrValue;
      ecgValue=analogRead(ECG);
      ecg_sum += ecgValue;
      emgValue=analogRead(EMG);
      emg_sum += emgValue;
      delayMicroseconds(500);
    }
  gsr_average = gsr_sum/10;
  ecg_average = ecg_sum/10;
  emg_average = emg_sum/10;
}

void sendData(){
  uint16_t sendSize = 0;
  
  dataPacket.timestamp = millis();
  dataPacket.ECG_data = gsr_average;
  dataPacket.GSR_data = ecg_average;
  dataPacket.EMG_data = emg_average;
  
  sendSize = myTransfer.txObj(dataPacket, sendSize);
  // Send buffer
  myTransfer.sendData(sendSize);
}
