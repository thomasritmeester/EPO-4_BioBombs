#include "SerialTransfer.h"
#include "I2Cdev.h"
#include "MPU6050.h"
#include <Adafruit_MLX90614.h>

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
#include "Wire.h"
#endif

MPU6050 mpu1;
MPU6050 mpu2(0x69);
Adafruit_MLX90614 tempSensor = Adafruit_MLX90614();
SerialTransfer myTransfer;

int Fs = 200; //Hz
unsigned long prevMillis;

const int ECG = A0;
const int GSR = A1;
const int EMG = A2;
//ACC sensor is connected to A4 and A5 using the I2C protocol

uint16_t ecgValue = 0;
uint16_t gsrValue = 0;
uint16_t emgValue = 0;

uint16_t gsr_average = 0;
uint16_t ecg_average = 0;
uint16_t emg_average = 0;

int16_t ax1, ay1, az1, gx1, gy1, gz1;
int16_t ax2, ay2, az2, gx2, gy2, gz2;

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

void setup()
{
  // join I2C bus (I2Cdev library doesn't do this automatically)
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif

  Serial.begin(57600);
  myTransfer.begin(Serial);
  tempSensor.begin();

  //initialize accelerometer
  mpu1.initialize();
  mpu1.setXGyroOffset(-2021);
  mpu1.setYGyroOffset(-817);
  mpu1.setZGyroOffset(1195);
  
  mpu2.initialize();
  mpu2.setXGyroOffset(-3527);
  mpu2.setYGyroOffset(-1899);
  mpu2.setZGyroOffset(519);
  
  mpu1.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  mpu2.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
}

void loop()
{
  takeMeasurements();
  
  // hold the program until 1/Fs has passed since the previous data packet was sent
  while (1) {
    if ((millis() - prevMillis) >= (1000 / Fs)) {
      prevMillis = millis();
      break;
    }
  }
  //send the actual data
  sendData();
}


void takeMeasurements() {
  //take 10 measurements and average them
  int ecg_sum = 0, gsr_sum = 0, emg_sum = 0;
//  for (int i = 0; i < 5; i++)
//  {
//    gsrValue = analogRead(GSR);
//    gsr_sum += gsrValue;
//    ecgValue = analogRead(ECG);
//    ecg_sum += ecgValue;
//    emgValue = analogRead(EMG);
//    emg_sum += emgValue;
//  }
//  gsr_average = gsr_sum;
//  ecg_average = ecg_sum;
//  emg_average = emg_sum;

  gsr_average = analogRead(GSR);
  ecg_average = analogRead(ECG);
  emg_average = analogRead(EMG);

  //measure accelerometer values
  mpu1.getMotion6(&ax1, &ay1, &az1, &gx1, &gy1, &gz1);
  mpu2.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);
  temperature = tempSensor.readObjectTempC()*100;
}

void sendData() {
  uint16_t sendSize = 0;

  dataPacket.timestamp = millis();
  dataPacket.ECG_data = gsr_average;
  dataPacket.GSR_data = ecg_average;
  dataPacket.EMG_data = emg_average;
  dataPacket.ACC1_X_data = ax1;
  dataPacket.ACC1_Y_data = ay1;
  dataPacket.ACC1_Z_data = az1;
  dataPacket.ACC2_X_data = ax2;
  dataPacket.ACC2_Y_data = ay2;
  dataPacket.ACC2_Z_data = az2;
  dataPacket.TEMP_DATA = temperature;

  sendSize = myTransfer.txObj(dataPacket, sendSize);
  // Send buffer
  myTransfer.sendData(sendSize);
}
