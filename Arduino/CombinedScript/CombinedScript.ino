#include "SerialTransfer.h"

SerialTransfer myTransfer;

long

const int ECG=A0;
const int GSR=A1;
uint16_t ecgValue=0;
uint16_t gsrValue=0;
uint16_t gsr_average=0;
uint16_t ecg_average=0;

struct STRUCT {
unsigned long timestamp;
uint16_t  ECG_data;
uint16_t GSR_data;
} dataPacket;

void setup()
{
  Serial.begin(57600);
  myTransfer.begin(Serial);
}

void loop()
{
//take 10 measurements and average them
long ecg_sum=0, gsr_sum=0, HR=0;
for(int i=0;i<10;i++)           
  {
    gsrValue=analogRead(GSR);
    gsr_sum += gsrValue;
    ecgValue=analogRead(ECG);
    ecg_sum += ecgValue;
    delayMicroseconds(500);
  }
gsr_average = gsr_sum/10;
ecg_average = ecg_sum/10;

uint16_t sendSize = 0;

dataPacket.timestamp = millis();
dataPacket.ECG_data = gsr_average;
dataPacket.GSR_data = ecg_average;

sendSize = myTransfer.txObj(dataPacket, sendSize);
///////////////////////////////////////// Send buffer
myTransfer.sendData(sendSize);
;
}
