//Script sampling the different values with different sample rates

#include "SerialTransfer.h"
#include "I2Cdev.h"
#include "Wire.h"
#include <Adafruit_MLX90614.h>

const int MPU_addr[] = {0x68, 0x69};
Adafruit_MLX90614 tempSensor = Adafruit_MLX90614();
SerialTransfer myTransfer;

const int ECG = A0;
const int GSR = A1;
const int EMG = A2;
//ACC sensor is connected to A4 and A5 using the I2C protocol

uint16_t ecg_value = 0;
uint16_t gsr_value = 0;
uint16_t emg_value = 0;

int16_t ax[] = {0,0};
int16_t ay[] = {0,0};
int16_t az[] = {0,0};

uint16_t temperature;

struct STRUCT {
  unsigned long timestamp;
  uint16_t  ECG_data;
  uint16_t GSR_data;
  uint16_t EMG_data;
  int16_t ACC1_X_data;
  int16_t ACC1_Y_data;
  int16_t ACC1_Z_data;
  int16_t ACC2_X_data;
  int16_t ACC2_Y_data;
  int16_t ACC2_Z_data;
  uint16_t TEMP_DATA;
} dataPacket;

void setup() {
  //starts serial connection
  Serial.begin(230400);
  myTransfer.begin(Serial);
  //initializes the temperature sensor
  tempSensor.begin();
//setup I2C transmission for accelerometer
  Wire.begin();
  for(int i=0; i<2; i++){   
    Wire.beginTransmission(MPU_addr[i]);
    Wire.write(0x6B);  // PWR_MGMT_1 register
    Wire.write(0);     // set to zero (wakes up the MPU-6050)
    Wire.endTransmission(true);
  }
}

float TsLow = 1000000/20.0; //us
float TsHigh = 1000000/700.0; //us
unsigned long prevMicrosHigh, prevMicrosLow = 0;

void loop() {
  if(micros() - prevMicrosLow > TsLow){
    prevMicrosLow = micros();
    takeMeasurementsLowFs();  
  }
  if(micros() - prevMicrosHigh > TsHigh){
    prevMicrosHigh = micros();
    takeMeasurementsHighFs();
    sendData();  
  }
}

void takeMeasurementsHighFs() {
  temperature = tempSensor.readObjectTempC()*100;
  ecg_value = analogRead(ECG);
  emg_value = analogRead(EMG);
}

void takeMeasurementsLowFs() {
  //measure accelerometer values
  for(int i=0; i<2; i++){
      Wire.beginTransmission(MPU_addr[i]);
      Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_addr[i], 6, true); // request a total of 6 registers
      int t = Wire.read();
      ax[i] = (t << 8) | Wire.read(); // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)
      t = Wire.read();
      ay[i] = (t << 8) | Wire.read(); // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
      t = Wire.read();
      az[i] = (t << 8) | Wire.read(); // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  }
  gsr_value = analogRead(GSR);
}

void sendData() {
  uint16_t sendSize = 0;

  dataPacket.timestamp = micros();
  dataPacket.ECG_data = gsr_value;
  dataPacket.GSR_data = ecg_value;
  dataPacket.EMG_data = emg_value;
  dataPacket.ACC1_X_data = ax[0];
  dataPacket.ACC1_Y_data = ay[0];
  dataPacket.ACC1_Z_data = az[0];
  dataPacket.ACC2_X_data = ax[1];
  dataPacket.ACC2_Y_data = ay[1];
  dataPacket.ACC2_Z_data = az[1];
  dataPacket.TEMP_DATA = temperature;

  sendSize = myTransfer.txObj(dataPacket, sendSize);
  // Send buffer
  myTransfer.sendData(sendSize);
}
