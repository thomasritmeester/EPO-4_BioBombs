#include "SerialTransfer.h"

SerialTransfer myTransfer;

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
// use this variable to keep track of how many
// bytes we’re stuffing in the transmit buffer
uint16_t sendSize = 0;

dataPacket.timestamp = millis();
dataPacket.ECG_data = 12345;
dataPacket.GSR_data = 23456;

sendSize = myTransfer.txObj(dataPacket, sendSize);
///////////////////////////////////////// Send buffer
myTransfer.sendData(sendSize);
delay(500);
}
