#!/usr/bin/env python3
########################################################################################################################
#
# author: Michel Daab
# date: 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of  this software and associated
# documentation files (the "Software"), to deal in  the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copied of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################################################################################################################

import os
import sys
import smbus
import math
import time
import statistics
import pickle
import heapq
import numpy as np

AXIS_16BIT_MAX_RANGE   = 32760.0### axis values are in [-32760 32760] for 16-bit output(.0 to use it as a float divisor)
AXIS_14BIT_MAX_RANGE    = 8192.0### axis values are in [-8192 8192] for 14-bit output(.0 to use it as a float divisor)
MAG_MAX_RANGE           = 4912.0### magnetic flux density values are in [-4912 4912](.0 to use it as a float divisor)

####################
### MPU9250/6500 ###
####################
MPU_SLAVE_ADDRESS       = 0x68  ### MPU9250/6500 Default I2C slave address
MPU_REG_WHO_AM_I        = 0x75  ### WHOAMI register
MPU_DEVICE_ID           = 0x71  ### value that should be read from WHOAMI

MPU_REG_INT_ENABLE      = 0x38  ### register to activate the AK8963 device
#MPU_ENABLING_RAW_SENSOR = 0x01  ### value to send to the INT_ENABLE register to activate the AK8963 device...
                                ### ...it enables Raw Sensor Data Ready interrupt to propagate to interrupt pin
MPU_REG_INT_PIN_CFG     = 0x37  ### int pin register

MPU_REG_PWR_MGMT_1      = 0x6B  ### power management register 1
MPU_REG_PWR_MGMT_2      = 0x6C  ### power management register 2
MPU_REG_I2C_MST_CTRL    = 0x24  ### I2C master control register
MPU_REG_CONFIG          = 0x1A  ### configuration register
MPU_REG_ACCEL_CONFIG    = 0x1C  ### accelerometer configuration register
MPU_REG_ACCEL_CONFIG_2  = 0x1D  ### accelerometer configuration register 2
MPU_REG_GYRO_CONFIG     = 0x1B  ### gyroscope configuration register

MPU_SET_DLPF            = 0x03  ### value to send to the configuration register to use the Digital Low Pass Filter(DLPF)
MPU_REG_SMPLRT_DIV      = 0x19  ### sample rate divider register
MPU_REG_INT_STATUS      = 0x3A  ### Interrupt Status register
MPU_REG_ACCEL_OUT       = 0x3B  ### accelerometer measurements register (6 registers back to back, MSB + LSB for x,y,z)
MPU_REG_GYRO_OUT        = 0x43  ### gyroscope measurements register (6 registers back to back, MSB + LSB for x,y,z)
MPU_REG_TEMP_OUT        = 0x41  ### temperature measurements register (2 registers back to back, MSB + LSB)

MPU_REG_USER_CTRL       = 0x6A  ### user control register
MPU_FIFO_EN             = 0x40  ### enable fifo mode (to send to MPU_REG_USER_CTRL)
MPU_I2C_MST_EN          = 0x20  ### Enable the I2C Master I/F module (to send to MPU_REG_USER_CTRL)

MPU_REG_FIFO_ENABLE     = 0x23  ### fifo enable register
MPU_FIFO_COUNT          = 0x72  ### FIFO Count Register (highs bits, the next address is the low bits)
MPU_FIFO_READ           = 0x74  ### FIFO read write register

MPU_REG_X_OFFS_USR_H    = 0x13  ### Gyro Offset Register, X high byte. Followed by the low byte register
                                ### and the regiters for y and z (6 registers in total)

MPU_REG_XA_OFFSET_H     = 0x77  ### AccelerometerOffset Register for x, High byte, the next register is the low byte
MPU_REG_YA_OFFSET_H     = 0x7A  ### AccelerometerOffset Register for y, High byte, the next register is the low byte
MPU_REG_ZA_OFFSET_H     = 0x7D  ### AccelerometerOffset Register for z, High byte, the next register is the low byte

MPU_REG_I2C_SLV0_ADDR   = 0x25  ### slave (compass) address register
MPU_REG_I2C_SLV0_REG    = 0x26  ### I2C slave 0 register address from where to begin data transfer
MPU_REG_I2C_SLV0_CTRL   = 0x27  ### control register for the slave 0
MPU_REG_I2C_SLV0_DO     = 0x63  ### slave data out register

##############
### AK8963 ###
##############
AK_SLAVE_ADDRESS    = 0x0C  ### AK8963 DEFAULT I2C slave address
AK_REG_WHO_AM_I     = 0x00  ### WHOAMI register
AK_DEVICE_ID        = 0x48  ### value that should be read from WHOAMI
AK_REG_CNTL1        = 0x0A  ### control register 1
AK_REG_CNTL2        = 0x0B  ### control register 2
AK_POWER_DOWN       = 0x00  ### value to send to control register 1 to power down the compass
AK_FUSE_MODE        = 0x0F  ### value to send to control register 1 to set the fuse mode
AK_REG_ASAX         = 0x10  ### Sensitivity Adjustment values
AK_REG_STATUS_1     = 0x02  ### status register 1
AK_REG_MAGNET_OUT   = 0x03  ### measurement data (6 registers back to back, LSB + MSB for x,y,z,
                            ### +1 register for Data overflow bit 3 and data read error status bit 2)
AK_EXT_SENS_DATA_00 = 0x49  ### external sensor data register

compass_calibration_file_name = "compass_calibration.dat"
calibrated_compass_file_name = "calibrated_compass.dat"

class Point3D:
    def __init__(self, x=0, y=0, z=0):
        """create a 3D vector
        :param x: x value of the point
        :type x: float
        :param y: y value of the point
        :type y: float
        :param z: z value of the point
        :type z: float"""
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_bytes_list(cls, bytes_list, msb_first=True, signed=True):
        """Initialize a Point3D instance from a bytes_list.
        :param bytes_list: a list of bytes. format: [byte_x_1,...,byte_x_n,byte_y_1,...,byte_y_n,byte_z_1,... byte_z_n]
        the size of the list must be a multiple of 3
        :type bytes_list: list
        :param msb_first: True if the most significant byte is first
        :type msb_first: bool
        :param signed: True if the value are signed
        :type signed: bool
        :return: a Point3D instance
        :rtype: Point3D"""
        if msb_first:
            str_byteorder = 'big'
        else:
            str_byteorder = 'little'
        third = len(bytes_list)//3
        x = int.from_bytes(bytes_list[:third], byteorder=str_byteorder, signed=signed)
        y = int.from_bytes(bytes_list[third:2 * third], byteorder=str_byteorder, signed=signed)
        z = int.from_bytes(bytes_list[2 * third:], byteorder=str_byteorder, signed=signed)
        return Point3D(x, y, z)

    def __add__(self, point_3d):
        """override the "+" operator
        :param point_3d: the point to add
        :type point_3d: Point3D
        :return: this point added to the other point
        :rtype: Point3D"""
        return Point3D(self.x + point_3d.x, self.y + point_3d.y, self.z + point_3d.z)

    def __iadd__(self, point_3d):
        """
        override the "+=" operator
        :param point_3d: the point to add
        :type point_3d: Point3D
        :return: this point added to the other point
        :rtype: Point3D
        """
        self.x += point_3d.x
        self.y += point_3d.y
        self.z += point_3d.z
        return self

    def __sub__(self, point_3d):
        """
        override the "-" operator
        :param point_3d: the point3D to subtract
        :type point_3d: Point3D
        :return: this point minus the other point
        :rtype: Point3D
        """
        return Point3D(self.x - point_3d.x, self.y - point_3d.y, self.z - point_3d.z)

    def __isub__(self, point_3d):
        """
        override the "-=" operator
        :param point_3d: the point to add
        :type point_3d: Point3D
        :return: this point subtract to the other point
        :rtype: Point3D
        """
        self.x -= point_3d.x
        self.y -= point_3d.y
        self.z -= point_3d.z
        return self

    def __mul__(self, multiplier):
        """
        override the "*" operator
        :param multiplier: the multiplier
        :type: float
        :return: this point multiplied by the multiplier
        :rtype: Point3D
        """
        return Point3D(self.x * multiplier, self.y * multiplier, self.z * multiplier)

    def __imul__(self, multiplier):
        """
        override the "*=" operator
        :param multiplier: the multiplier
        :type: float
        :return: this point multiplied by the multiplier
        :rtype: Point3D
        """
        self.x *= multiplier
        self.y *= multiplier
        self.z *= multiplier
        return self

    def __truediv__(self, divider):
        """
        override the "/" operator
        :param divider: the divider
        :type divider: float
        :return: this point divided by the divider
        :rtype: Point3D
        """
        return Point3D(self.x / divider, self.y / divider, self.z / divider)

    def __neg__(self):
        """override the "negative" operator (the "-" in front of a value, unary operator)
        :return: "-this point"
        :rtype: Point3D"""
        return Point3D(-self.x, -self.y, -self.z)

    def __pow__(self, power):
        """override the "**" operator
        :param power: the power value
        :type power: float
        :return: this point raised at the given power
        :rtype: Point3D"""
        return Point3D(self.x ** power, self.y ** power, self.z ** power)

    def length(self):
        """Return the length of the vector defined by this point
        :return: the length of the vector
        :rtype: float"""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def as_list(self):
        """return the 3 attributes as a list
        :return: [x, y, z]
        :rtype: list"""
        return [self.x, self.y, self.z]

    def dot(self, point_3d):
        """Return the dot product between this point and another
        :param point_3d: the other point of the dot product
        :type point_3d: Point3D
        :return: dot product
        :rtype: float"""
        return self.x * point_3d.x + self.y * point_3d.y + self.z * point_3d.z

    def cross(self, point_3d):
         """Return the cross product between this point and another
         :param point_3d: the other point of the cross product
         :type point_3d: Point3D
         :return: cross product
         :rtype: Point3D"""
         return Point3D(self.y * point_3d.z - self.z * point_3d.y,
                        self.z * point_3d.x - self.x * point_3d.z,
                        self.x * point_3d.y - self.y * point_3d.x)

    def multiply_attribute_by_attribute(self, point_3d_multiplier):
        """ return the product of each attribute: self.x by point_3d.x and so on
        :param point_3d_multiplier: a vector of 3 multipliers
        :type point_3d_multiplier: Point3D
        :return: this Point3D with each attribute multiplied
        :rtype: Point3D"""
        self.x *= point_3d_multiplier.x
        self.y *= point_3d_multiplier.y
        self.z *= point_3d_multiplier.z
        return self

    def to_bytes_list(self, msb_first=True):
        """turn the x,y,z values into a list of 6 bytes, 2 for each value. If x,y,z are not integers they are rounded first
        :param msb_first: True if the most significant byte is first for each value
        :type msb_first: bool
        :return: list of 6 bytes for x, y, z
        :rtype: list"""
        if msb_first:
            str_byteorder = 'big'
        else:
            str_byteorder = 'little'
        return list(round(self.x).to_bytes(2, byteorder=str_byteorder, signed=True)
                    + round(self.y).to_bytes(2, byteorder=str_byteorder, signed=True)
                    + round(self.z).to_bytes(2, byteorder=str_byteorder, signed=True))

    def __str__(self):
        """:return: a string listing the values of the instance
        :rtype: string"""
        return "x: {}, y: {}, z: {}".format(round(self.x, 3), round(self.y, 3), round(self.z, 3))

    @staticmethod
    def squared_distance(point3d_1, point3d_2=None):
        """
        return the squared value of the 2-norm distance between 2 points, or the size of the first 3D point
        :param point3d_1: the 3D point we measure the distance from
        :type point3d_1: Point3D
        :param point3d_2: the 3D point we measure the distance to. If ignored, return the size of the point3D_1 instead
        :type point3d_2: Point3D
        :return: the distance between 2 points or the size of a 3D point
        :rtype: float
        """
        if point3d_2:
            point3d_1 -= point3d_2
        return point3d_1.x ** 2 + point3d_1.y ** 2 + point3d_1.z ** 2

    @staticmethod
    def mean_squared_error(point3d_evaluated_list, point3d_real_list=None):
        """
        return the mean squared error between a list of evaluated 3D points and the list of values those points
        should have
        :param point3d_evaluated_list: list of 3D points as evaluated
        :type: list
        :param point3d_real_list: list of real point3D, if None then the MSE will be calculated from (0, 0, 0),
        must be of the same size than the point3d_evaluated_list
        :type: list
        :return: the mean squared error
        """
        ### create a list of distance between evaluated points and corresponding ones, divide by 3 because a 3D point
        ### has 3 dimensions, then do a mean of those values
        if point3d_real_list:
            return statistics.mean([Point3D.squared_distance(evaluated, real)/3
                                    for evaluated, real in zip(point3d_evaluated_list, point3d_real_list)])
        else:
            return statistics.mean([Point3D.squared_distance(evaluated)/3 for evaluated in point3d_evaluated_list])

class I2C:
    """main class for I2C devices"""
    def __init__(self, i2c_number, address):
        self.bus = smbus.SMBus(i2c_number)  ### "i2c_number" took from-> /dev/i2c-"i2c_number"
        self.address = address

    def read(self, register):
        """read a data from the device"""
        return self.bus.read_byte_data(self.address, register)

    def read_block(self, register, block_length):
        """read a block of byte from the given register, return a list of bytes"""
        return self.bus.read_i2c_block_data(self.address, register, block_length)

    def write(self, register, value):
        self.bus.write_byte_data(self.address, register, value)

    def write_block(self, register, bytes_list):
        """write a list of bytes"""
        self.bus.write_i2c_block_data(self.address, register, bytes_list)

    @staticmethod
    def two_bytes_to_int(msb, lsb, signed=True):
        """convert 2 bytes into a 16-bit int
        msb: most significant byte
        lsb: least significant byte
        signed: true if we want to return a signed integer"""
        return int.from_bytes([msb, lsb], byteorder='big', signed=signed)

    @staticmethod
    def int_to_two_bytes(integer, signed=True):
        """convert an int into 2 bytes
        integer: integer to convert
        signed: true if the integer is signed

        return the most significant byte first"""
        return integer.to_bytes(2, byteorder='big', signed=signed)


class MPU6500(I2C):
    def __init__(self, i2c_number):
        super().__init__(i2c_number, MPU_SLAVE_ADDRESS)
        self._delay_to_next_sample = 0  ### time to wait before getting the next sample value
        self._sample_rate = 0

        self._acce_resolution = 1.0
        self._acce_current_range = 8
        self._acce_current_dlpf_selector = 2

        self._gyro_resolution = 1.0
        self._gyro_current_range = 500
        self._gyro_current_dlpf_selector = 2

        self._fifo_acce = False     ### are the acceleration measures send to the fifo
        self._fifo_temp = False     ### are the temperature measures send to the fifo
        self._fifo_gyro = False     ### are the gyroscope measures send to the fifo
        self._fifo_buffer_size = 0  ### size in bytes of the fifo buffer

    def check_device(self):
        """test whether it's a true device"""
        return MPU_DEVICE_ID == self.read(MPU_REG_WHO_AM_I)

    def enable_i2c_master_mode(self):
        """Enable I2C master mode for compass"""
        self.write(MPU_REG_USER_CTRL, (self.read(MPU_REG_USER_CTRL) | 0x20))

    def set_sample_rate(self, rate):
        """Set sampling rate to the closest value between 4 and 1000Hz
        :param rate: the requested rate
        :type rate: int"""
        if rate < 4:
            rate = 4
        elif rate > 1000:
            rate = 1000

        divider = 1000 // rate - 1  ### "-1" because the MPU automatically add +1 (see datasheet)
        self.write(MPU_REG_SMPLRT_DIV, divider)
        self._sample_rate = 1000 / divider
        self._delay_to_next_sample = 0.001 * divider

    def set_acce_range(self, acce_range):
        """set the range of the accelerometer
        acce_range : 2/4/8/16 g"""
        self._acce_current_range = acce_range
        acce_range_dict = {2: 0x00,
                           4: 0x08,
                           8: 0x10,
                           16: 0x18}
        self.write(MPU_REG_ACCEL_CONFIG, acce_range_dict[acce_range])
        self._acce_resolution = acce_range / AXIS_16BIT_MAX_RANGE

    def set_acce_dlpf(self, acce_dlpf_selector):
        """set the Digital Low Pass Filter bandwidth for the accelerometer
        acce_dlpf_selector : the Digital Low Pass Filter bandwidth for the accelerator
                            0 (0x00) = 218.1Hz
                            1 (0x01) = 218.1Hz
                            2 (0x02) =  99Hz
                            3 (0x03) =  44.8Hz
                            4 (0x04) =  21.2Hz
                            5 (0x05) =  10.2Hz
                            6 (0x06) =   5.05Hz
                            7 (0x07) = 420Hz"""
        self._acce_current_dlpf_selector = acce_dlpf_selector
        self.write(MPU_REG_ACCEL_CONFIG_2, acce_dlpf_selector)

    def set_gyro_range(self, gyro_range):
        """set the scale of the gyroscope
        gyro_range : 250/500/1000/2000 degrees by second (dps)"""
        self._gyro_current_range = gyro_range
        gyro_range_dict = {250: 0x00,
                           500: 0x08,
                           1000: 0x10,
                           2000: 0x18}

        self.write(MPU_REG_GYRO_CONFIG, gyro_range_dict[gyro_range])
        self._gyro_resolution = gyro_range / AXIS_16BIT_MAX_RANGE

    def set_gyro_dlpf(self, gyro_dlpf_selector):
        """set the Digital Low Pass Filter bandwidth for the gyroscope
        gyro_dlpf_selector : the Digital Low Pass Filter bandwidth for the gyroscope
                            0 (0x00) =  250Hz
                            1 (0x01) =  184Hz
                            2 (0x02) =   92Hz
                            3 (0x03) =   41Hz
                            4 (0x04) =   20Hz
                            5 (0x05) =   10Hz
                            6 (0x06) =    5Hz"""
        self._gyro_current_dlpf_selector = gyro_dlpf_selector
        self.write(MPU_REG_CONFIG, gyro_dlpf_selector)

    def is_ready(self):
        """return 1 if the data are ready to be read, 0 otherwise"""
        return self.read(MPU_REG_INT_STATUS) & 0x01

    def get_raw_acceleration(self):
        return Point3D.from_bytes_list(self.read_block(MPU_REG_ACCEL_OUT, 6))

    def get_raw_gyroscope(self):
        return Point3D.from_bytes_list(self.read_block(MPU_REG_GYRO_OUT, 6))

    def get_acceleration(self):
        return self.get_raw_acceleration() * self._acce_resolution

    def get_gyroscope(self):
        return self.get_raw_gyroscope() * self._gyro_resolution

    def get_temperature(self):
        data = self.read_block(MPU_REG_TEMP_OUT, 2)
        temp = self.two_bytes_to_int(data[0], data[1])
        return (temp - 21.0) / 333.87 + 21.0    ### from datasheets

    def reset_device(self):
        """Reset the internal registers and restores the default settings"""
        self.write(MPU_REG_PWR_MGMT_1, 0x80)    ### reset the device
        time.sleep(0.1)
        self.write(MPU_REG_PWR_MGMT_1, 0x00)    ### wake up the device

    def set_fifo(self, acce=True, temp=True, gyro=True):
        """enable or disable the fifo buffers
        input: True or False to enable their respective buffer"""
        self._fifo_acce = acce
        self._fifo_temp = temp
        self._fifo_gyro = gyro
        self._fifo_buffer_size = temp * 2 + gyro * 6 + acce * 6

        self.write(MPU_REG_USER_CTRL, MPU_FIFO_EN | MPU_I2C_MST_EN)
        self.write(MPU_REG_FIFO_ENABLE,
                             (acce * 0x08) |    ### value to enable the gyro buffer
                             (temp * 0x80) |    ### value to enable the temp buffer
                             (gyro * 0x70))     ### value to enable the gyro buffer

    def calibrate(self, calibration_time=1):
        """calibrate the gyroscope and the accelerometer
        :param calibration_time: time over which the mean value of the bias will be got (in seconds)
        :type calibration_time: int"""
        self.reset_device()

        ####### get stable time source
        self.write(MPU_REG_PWR_MGMT_1, 0x01)
        self.write(MPU_REG_PWR_MGMT_2, 0x00)
        time.sleep(0.2)

        ####### configure device for bias calculation
        self.write(MPU_REG_INT_ENABLE, 0x00)    ### disable all interrupts
        self.write(MPU_REG_FIFO_ENABLE, 0x00)   ### disable FIFO
        self.write(MPU_REG_I2C_MST_CTRL, 0x00)  ### disable I2C master
        self.write(MPU_REG_USER_CTRL, 0x0C)     ### disable FIFO, I2C master mode, reset FIFO and DMP
        time.sleep(0.1)

        ####### configure gyroscope and accelerometer for bias calculation
        self.set_gyro_dlpf(1)      ### set gyroscope DLPF to 184
        self.set_sample_rate(100)  ### set sample rate to 100Hz (1 sample every each 0.01s)
        self.set_gyro_range(250)   ### set the gyroscope range to 250, the maximum sensitivity
        self.set_acce_range(2)     ### Set accelerometer full-scale to 2 g, maximum sensitivity

        ####### Configure FIFO to capture accelerometer and gyro data for bias calculation
        self.write(MPU_REG_USER_CTRL, 0x40)  ### Enable FIFO
        self.set_fifo(acce=True, temp=False, gyro=True)  ### Enable gyro and accelerometer sensors for FIFO

        ####### mean values over calibration_time
        acce_bias_3d = Point3D()
        gyro_bias_3d = Point3D()

        count = 0
        start = time.time()
        while time.time() - start < calibration_time:
            count += 1
            time.sleep(self._delay_to_next_sample)
            acce_bias_3d += self.get_raw_acceleration()
            gyro_bias_3d += self.get_raw_gyroscope()
        acce_bias_3d = acce_bias_3d / count
        gyro_bias_3d = gyro_bias_3d / count

        acce_gravity = Point3D(0, 0, -1 / self._acce_resolution)  ### taking into account Earth gravity (doesn't work
                                                                  ### if the calibration is done on another planet!)
        acce_bias_3d = acce_bias_3d - acce_gravity  ### the gravity shouldn't be counted as a bias

        ####### set the bias to the hardware
        self.set_acce_bias_to_hardware(acce_bias_3d)
        self.set_gyro_bias_to_hardware(gyro_bias_3d)    ### set gyro bias to hardware

    def set_gyro_bias_to_hardware(self, gyro_bias_3d):
        """set the gyro bias to the hardware gyro bias registers
        :param gyro_bias_3d: a Point3D containing the gyroscope bias
        :type gyro_bias_3d: Point3D"""
        gyro_bias_3d *= self._gyro_current_range / 1000     ### not sure why it works, maybe something with the
                                                            ### offsetDPS in the offset register
        old_gyro_bias = Point3D.from_bytes_list(self.read_block(MPU_REG_X_OFFS_USR_H, 6))  ### previous gyro bias
        self.write_block(MPU_REG_X_OFFS_USR_H, (old_gyro_bias - gyro_bias_3d).to_bytes_list())

    def set_acce_bias_to_hardware(self, acce_bias_3d):
        """set the accelerometer bias to the hardware accelermoter bias registers
        :param acce_bias_3d: a Point3D containing the accelerometer bias
        :type acce_bias_3d: Point3D"""
        old_acce_bias = Point3D.from_bytes_list(self.read_block(MPU_REG_XA_OFFSET_H, 2)
                                                + self.read_block(MPU_REG_YA_OFFSET_H, 2)
                                                + self.read_block(MPU_REG_ZA_OFFSET_H, 2))
        new_acce_bias = old_acce_bias - acce_bias_3d * self._acce_current_range / 16
        msb_x, lsb_x, msb_y, lsb_y, msb_z, lsb_z = new_acce_bias.to_bytes_list()

        self.write_block(MPU_REG_XA_OFFSET_H, [msb_x, lsb_x])
        self.write_block(MPU_REG_YA_OFFSET_H, [msb_y, lsb_y])
        self.write_block(MPU_REG_ZA_OFFSET_H, [msb_z, lsb_z])

    def read_fifo(self):
        """read the buffer
        :returns: averaged raw values from the accelerometer, temperature and gyroscope
        :rtype: Point3D, float, Point3D"""
        written_bytes_count = I2C.two_bytes_to_int(*self.read_block(MPU_FIFO_COUNT, 2))
        read_count = int(written_bytes_count / self._fifo_buffer_size)  ### how many times the fifo buffer will be read

        acce_raw_3d = Point3D()
        temp_raw = 0
        gyro_raw_3d = Point3D()

        for _ in range(read_count):
            buffer_data = self.read_block(MPU_FIFO_READ, self._fifo_buffer_size)

            if self._fifo_acce:  ### accelerometer data
                acce_raw_3d += Point3D.from_bytes_list(buffer_data[:6])

            if self._fifo_temp:  ### temperature data
                offset = self._fifo_acce * 6
                temp_raw += I2C.two_bytes_to_int(buffer_data[0 + offset], buffer_data[1 + offset])

            if self._fifo_gyro:  ### gyroscope data
                offset = self._fifo_acce * 6 + self._fifo_temp * 2
                gyro_raw_3d += Point3D.from_bytes_list(buffer_data[offset:6 + offset])

        return acce_raw_3d / read_count, temp_raw / read_count, gyro_raw_3d / read_count


# class AK8963(I2C):
#     def __init__(self, i2c_number):
#         super().__init__(i2c_number, AK_SLAVE_ADDRESS)
#         self._coef_3d = Point3D()
#         self._resolution = 0
#
#     def check_device(self):
#         """test whether it's a true device"""
#         return AK_DEVICE_ID == self.read(AK_REG_WHO_AM_I)
#
#     def config(self, mode=100, precision=16):
#         """configure the compass
#         mode : 8 or 100 Hz (0x02) or 100 Hz (0x06) compass data ODR
#         precision : 14 or 16 (bit)"""
#         self.write(AK_REG_CNTL1, AK_POWER_DOWN)  ### compass power down
#         time.sleep(0.01)
#         self.write(AK_REG_CNTL1, AK_FUSE_MODE)   ### setting fuse mode
#         time.sleep(0.01)
#         self.set_precision(mode, precision)
#         # self.write(AK_REG_CNTL1, AK_POWER_DOWN)  ### compass power down
#         time.sleep(0.01)
#
#     def set_precision(self, mode, precision):
#         """set the precision of the compass
#         mode : 8 or 100 Hz (0x02) or 100 Hz (0x06) compass data ODR
#         precision : 14 or 16 (bit)"""
#         if precision == 14:
#             self._resolution = MAG_MAX_RANGE / AXIS_14BIT_MAX_RANGE
#             precision_bit = 0x00
#         else:
#             self._resolution = MAG_MAX_RANGE / AXIS_16BIT_MAX_RANGE
#             precision_bit = 0x10
#         mode_dict = {8: 0x02,
#                      100: 0x06}
#         self.write(AK_REG_CNTL1, (mode_dict[mode] | precision_bit))
#
#         ### Read the x, y, and z-axis calibration values
#         data = self.read_block(AK_REG_ASAX, 3)   ### read the ASAX register and the 2 following registers, ASAY and ASAZ
#         self._coef_3d = Point3D((data[0] - 128) / 256.0 + 1.0,
#                                 (data[1] - 128) / 256.0 + 1.0,
#                                 (data[2] - 128) / 256.0 + 1.0) * self._resolution
#
#     def is_ready(self):
#         """return 1 if the data are ready to be read, 0 otherwise"""
#         return self.read(AK_REG_STATUS_1)   ### only the least significant bit can change, all the other are always 0
#
#     def get_raw_compass(self):
#         return Point3D.from_bytes_list(self.read_block(AK_REG_MAGNET_OUT, 6), msb_first=False)

### MPU9250 I2C Control class
class MPU9250(MPU6500):
    """Control a MPU9250 device with a I2C bus, which is a gyro/accelerometer MPU6500 device(main device)
    and a AK8963 compass (secondary device)"""
    def __init__(self, i2c_number=1):
        super().__init__(i2c_number)

        self._compass_resolution = 0
        self._compass_precision = 0
        self._compass_sample_rate = 8
        self._compass_delay_to_next_sample = 0.125
        self._compass_calibration_3d = Point3D()
        self.activate_ak8963()  ### activate compass
        self.calibration_file_name = compass_calibration_file_name   ### path/name to save the compass calibration data
        self.calibrated_file_name = calibrated_compass_file_name     ### path/name to save the calibrated compass data
        self.compass_correction_matrix = np.identity(3)
        self.compass_correction_vector = Point3D()

    def write_to_compass(self, register, value):
        self.write(MPU_REG_I2C_SLV0_ADDR, AK_SLAVE_ADDRESS)  ### Set the I2C slave address of AK8963 and set for write.
        self.write(MPU_REG_I2C_SLV0_REG, register)
        self.write(MPU_REG_I2C_SLV0_DO, value)
        self.write(MPU_REG_I2C_SLV0_CTRL, 0x81)  ### Enable I2C and write 1 byte
        time.sleep(self._compass_delay_to_next_sample)

    def read_block_from_compass(self, register, block_length):
        self.write(MPU_REG_I2C_SLV0_ADDR, AK_SLAVE_ADDRESS | 0x80)  ### Set the I2C slave and set for read.
        self.write(MPU_REG_I2C_SLV0_REG, register)
        self.write(MPU_REG_I2C_SLV0_CTRL, 0x80 | block_length)  ### Enable I2C
        time.sleep(self._compass_delay_to_next_sample)
        return self.read_block(AK_EXT_SENS_DATA_00, block_length)

    def get_raw_compass(self):
        data = self.read_block_from_compass(AK_REG_MAGNET_OUT, 7)
        return Point3D.from_bytes_list(data[:6], msb_first=False)

    def set_compass_precision(self, mode=100, precision=16):
        """set the mode and the precision of the compass
        :param mode: 8 or 100 Hz (0x02) or 100 Hz (0x06) compass data ODR
        :type mode: int
        :param precision: precision on 14 or 16 bits
        :type precision: int"""
        if precision == 14:
            self._compass_resolution = MAG_MAX_RANGE / AXIS_14BIT_MAX_RANGE
            precision_bit = 0x00
        else:
            self._compass_resolution = MAG_MAX_RANGE / AXIS_16BIT_MAX_RANGE
            precision_bit = 0x10

        self._compass_precision = precision
        self._compass_sample_rate = mode
        self._compass_delay_to_next_sample = 1 / mode

        mode_dict = {8: 0x02,
                     100: 0x06}

        self.write_to_compass(AK_REG_CNTL1, (mode_dict[mode] | precision_bit))

    def activate_ak8963(self):
        """Enabling of the AK8963 device"""
        self.write_to_compass(AK_REG_CNTL1, 0x01)     ### Reset AK8963
        self.write_to_compass(AK_REG_CNTL1, 0x00)     ### power down compass
        self.write_to_compass(AK_REG_CNTL1, 0x01)     ### enter fuse mode
        data = self.read_block_from_compass(AK_REG_ASAX, 3)  ### read 3 bytes, starting from AK_REG_ASAX
        self._compass_calibration_3d = Point3D((data[0] - 128) / 256.0 + 1.0,
                                               (data[1] - 128) / 256.0 + 1.0,
                                               (data[2] - 128) / 256.0 + 1.0)
        self.write_to_compass(AK_REG_CNTL1, 0x00)  ### power down compass
        self.set_compass_precision()

    def create_calibrating_measure_for_compass(self, calibration_time=20, calibration_file_name=None):
        """
        :param calibration_time: time in second we take 3D points measures from the compass
        :type calibration_time: float
        :param calibration_file_name: the path and name to the file to save a pickle dump of a 3D points list
        of random compass measures
        :type calibration_file_name: str"""
        input("Move gradually the MPU in every directions for {} secs. "
              "Press any key when ready to proceed.".format(calibration_time))

        delay = 5
        for second in range(delay, 1, -1):
            print("start in {} seconds".format(second))
            time.sleep(1)
        print("start in 1 second")
        time.sleep(1)
        print("Move the MPU!")

        measure_list = []
        start = time.time()
        while time.time() - start < calibration_time:
            calibration_measure_3d = self.get_raw_compass()\
                .multiply_attribute_by_attribute(self._compass_calibration_3d) \
                * self._compass_resolution

            measure_list.append(calibration_measure_3d)

        if not calibration_file_name:                           ### if a file name is not defined,
            calibration_file_name = self.calibration_file_name  ### we use the one by default
        else:
            self.calibration_file_name = calibration_file_name  ### otherwise we store the new one

        with open(calibration_file_name, "wb") as calibration_data_file:
            pickle.dump(measure_list, calibration_data_file)

    def get_calibration(self, calibrated_file_name=None):
        """ Get the correction matrix and correction vector for the compass from the specified file or the default file.
        If the file doesn't exist create one.
        :param calibrated_file_name: the path and name to the file where the correction matrix and the correction
        vector are stored in.
        :type calibrated_file_name: str"""
        if not calibrated_file_name:    ### using the file by default if not provided
            calibrated_file_name = self.calibrated_file_name

        if os.path.isfile(calibrated_file_name):    ### get the matrix and the vector if the file exists
            with open(calibrated_file_name, "rb") as calibrated_compass_file:
                self.compass_correction_matrix, self.compass_correction_vector = pickle.load(calibrated_compass_file)
                return

        if not os.path.isfile(self.calibration_file_name):  ### if their is no calibration file,
            self.create_calibrating_measure_for_compass()              ### we create one

        ### calculating the matrix and the vector and storing it
        self.calibrating_compass(calibrated_file_name=calibrated_file_name)


    def get_best_compass_calibration(self, min_extrema_number=1, max_extrema_number=30,
                                     calibration_file_name=None, calibrated_file_name=None):
        """ Find and store the best compass calibration from a calibration file with a list of random measures and a
        range of extrema.
        :param min_extrema_number: start number of extrema to test
        :type: int
        :param max_extrema_number: end number of extrema to test
        :type: int
        :param calibration_file_name: the path and name to the file containing a pickle dump of a 3D points list
        of random compass measures
        :type calibration_file_name: str
        :param calibrated_file_name: the path and name to the file to store the correction matrix and the correction
        vector in.
        :type calibrated_file_name: str"""
        ### get the calibration file
        if not calibration_file_name:
            calibration_file_name = self.calibration_file_name

        try:
            with open(calibration_file_name, "rb") as calibration_data_file:
                calibration_measure_list = pickle.load(calibration_data_file)

        except FileNotFoundError:
            print("No calibration measures file found")
            return

        ### test all the calibrating to get the smallest error
        min_error = sys.maxsize
        best_soft_iron_bias_3_by_3 = None
        best_hard_iron_bias_3d = None
        for extrema_number in range(min_extrema_number, max_extrema_number + 1):
            print("compass evaluation:", extrema_number - min_extrema_number + 1,
                  "of:", max_extrema_number - min_extrema_number + 1)
            soft_iron_bias_3_by_3, hard_iron_bias_3d = self.calibrating_compass(extrema_number,
                                                                                calibration_measure_list)
            error = self.evaluate_compass_calibration(calibration_measure_list, 
                                                      soft_iron_bias_3_by_3, 
                                                      hard_iron_bias_3d)
            if error < min_error:
                min_error = error
                best_soft_iron_bias_3_by_3 = soft_iron_bias_3_by_3
                best_hard_iron_bias_3d = hard_iron_bias_3d

        ### storing the best bias values
        self.compass_correction_matrix = best_soft_iron_bias_3_by_3
        self.compass_correction_vector = best_hard_iron_bias_3d

        if not calibrated_file_name:
            calibrated_file_name = self.calibrated_file_name

        with open(calibrated_file_name, "wb") as calibrated_file:
            pickle.dump([best_soft_iron_bias_3_by_3, best_hard_iron_bias_3d], calibrated_file)

    def evaluate_compass_calibration(self, calibration_measure_list, soft_iron_bias_3_by_3, hard_iron_bias_3d):
        """ return the mean squared error of the corrected measure (all corrected measures should be of length 1)
        :param calibration_measure_list: a lsit of Point3D measures
        :type calibration_measure_list: list
        :param soft_iron_bias_3_by_3: the 3*3 numpy matrix to correct soft iron
        :param full_process_matrix: np.array
        :param hard_iron_bias_3d: the vector to correct the hard iron bias
        :type hard_iron_bias_3d: Point3D
        :return: the mean squared error
        :rtype: float
        """
        error_dist = 0
        for calibration_measure_3d in calibration_measure_list:
            hard_iron_corrected_3d = calibration_measure_3d - hard_iron_bias_3d
            corrected_measure_3d = Point3D(*soft_iron_bias_3_by_3.dot((hard_iron_corrected_3d).as_list()))
            error_dist += (corrected_measure_3d.length() - 1) ** 2
        return error_dist / len(calibration_measure_list)

    def calibrating_compass(self, extrema_number, calibration_measure_list):
        """ Calculate a correction matrix and a correction vector for the hard iron and soft iron bias.
        :param extrema_number: the number of extrema to consider when looking for a right angle between potential axis
        :type extrema_number: int
        :param calibration_measure_list: a list of random measures
        :type calibration_measure_list: list"""
        ### step 1: define the bias of hard iron, i.e. the vector to center the ellipsoid on 0,0,0
        ### step 2: remove the hard iron bias from every point, to center the ellipsoid
        ### step 3: search the longuest and shortest vectors, should be the major and minor axis of the ellipsoid
        ### step 4: from those two axis we look for a third, orthogonal to the first two to havee a new basis
        ### step 5: create a new orthonormal basis based on the major and minor axis of the ellipsoid
        ### step 6: search from the 3d points the closest real points (angle wise) to that theoretical axis
        ### step 7: search the scales on each axis to turn the ellipsoid into a sphere with r == 1
        ### step 8: get the matrix to change to the new basis, change the scale and go back to the regular basis

        ### step 1: define the bias of hard iron, i.e. the vector to center the ellipsoid on 0,0,0
        x_list = [calibration_measure_3d.x for calibration_measure_3d in calibration_measure_list]
        min_x = min(x_list)
        max_x = max(x_list)
        y_list = [calibration_measure_3d.y for calibration_measure_3d in calibration_measure_list]
        min_y = min(y_list)
        max_y = max(y_list)
        z_list = [calibration_measure_3d.z for calibration_measure_3d in calibration_measure_list]
        min_z = min(z_list)
        max_z = max(z_list)

        hard_iron_bias_3d = Point3D((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)

        ### step 2: remove the hard iron bias from every point, to center the ellipsoid
        unbiased_measure_list = []
        for calibration_measure_3d in calibration_measure_list:
            unbiased_measure_list.append(calibration_measure_3d - hard_iron_bias_3d)

        ####### calculing the length to each point and store it so we don't have to calculate it again
        for calibration_measure_3d in unbiased_measure_list:
            calibration_measure_3d.len = calibration_measure_3d.length()

        ### step 3: search for the longuest and shortest point that form an angle closest to pi/2 to get a new
        ### orthogonalbasis, with the longuest being the major axis of the ellipsoid, the shortest the minor axis.  we
        ### use several points each time to eliminate possible outliers, and use the closests to
        ### cos == 0 ( angle == pi/2) to make the final selection
        extrema_list_size = extrema_number
        max_len_list = heapq.nlargest(extrema_list_size, unbiased_measure_list, key=lambda point_3d: point_3d.len)
        min_len_list = heapq.nsmallest(extrema_list_size, unbiased_measure_list, key=lambda point_3d: point_3d.len)

        min_cos = 1
        real_point_3d_axis_1 = None
        real_point_3d_axis_2 = None

        for point_3d_max in max_len_list:
            for point_3d_min in min_len_list:
                cos = point_3d_min.dot(point_3d_max) / (point_3d_min.len * point_3d_max.len)
                if abs(cos) < min_cos:
                    min_cos = abs(cos)
                    real_point_3d_axis_1 = point_3d_max
                    real_point_3d_axis_2 = point_3d_min

        ### step 4: from those two axis we look for a third, orthogonal to the first two to havee a new basis
        raw_theoretical_axis_3 = real_point_3d_axis_1.cross(real_point_3d_axis_2)

        ### step 5: create a new orthonormal basis based on the major and minor axis of the ellipsoid
        ### we create raw_theoretical_axis_2 because real_point_3d_axis_2 is probably not exactly perpendicular to
        ### real_point_3d_axis_1
        ### in the end, only real_point_3d_axis_1 and theoretical_axis_1 are exactly aligned, theoretical_axis_2 and
        ### theoretical_axis_3 are exactly perpendicular to it with real_point_3d_axis_2 and real_point_3d_axis_3 being
        ### the best real approximations
        theoretical_axis_1 = real_point_3d_axis_1 / real_point_3d_axis_1.length()
        raw_theoretical_axis_2 = raw_theoretical_axis_3.cross(real_point_3d_axis_1)
        theoretical_axis_2 = raw_theoretical_axis_2 / raw_theoretical_axis_2.length()
        theoretical_axis_3 = raw_theoretical_axis_3 / raw_theoretical_axis_3.length()

        ### step 6: search from the 3d points the closest real points (angle wise) to that theoretical axis
        ### search the point that makes the smallest angle with it,
        ### i.e. |cos| ~= 1
        raw_theoretical_axis_3.len = raw_theoretical_axis_3.length()
        max_cos = 0
        real_point_3d_axis_3 = None
        for point_3d in unbiased_measure_list:
            cos = raw_theoretical_axis_3.dot(point_3d) / (raw_theoretical_axis_3.len * point_3d.len)
            if abs(cos) > max_cos:
                max_cos = abs(cos)
                real_point_3d_axis_3 = point_3d

        ### step 7: based on the 3 points closest to the axis of the new basis, search the scale on each axis to turn
        ### the ellipsoid into a sphere with r == 1
        ### if everything went right, scale_vector_1 <= scale_vector_3 <= scale_vector_2 since vector 1 has been
        ### created from the longuest dimension of the ellipsoid, and vector 2 from the smallest
        scale_vector_1 = 1 / abs(real_point_3d_axis_1.length())
        scale_vector_2 = 1 / abs(real_point_3d_axis_2.length())
        scale_vector_3 = 1 / abs(real_point_3d_axis_3.length())

        ### step 8: combine the matrix to change to the new basis, change the scale and go back to the regular basis
        new_orthonormal_basis = np.array((theoretical_axis_1.as_list(),
                                          theoretical_axis_2.as_list(),
                                          theoretical_axis_3.as_list())).T

        change_scale_matrix = np.array(([scale_vector_1, 0, 0],
                                        [0, scale_vector_2, 0],
                                        [0, 0, scale_vector_3]))

        matrix_to_change_to_new_matrix = np.linalg.inv(new_orthonormal_basis)

        soft_iron_bias_3_by_3 = new_orthonormal_basis.dot(change_scale_matrix.dot(matrix_to_change_to_new_matrix))

        return soft_iron_bias_3_by_3, hard_iron_bias_3d


if __name__ == "__main__":
    mpu = MPU9250()
    mpu.calibrate(2)
    # mpu.calibrate_compass()

    mpu.get_best_compass_calibration()


