#!/usr/bin/env python
# coding=utf-8

import RPi.GPIO as gpio
import time

high = gpio.HIGH
low = gpio.LOW
out = gpio.OUT
pin = 11
on_time = 1

def open():
    gpio.setmode(gpio.BOARD)
    gpio.setup(pin, out)
    try:
        gpio.output(pin, high)
        time.sleep(on_time)
    except Exception:
        pass
    finally:
        gpio.cleanup()

if __name__ == '__main__':
    open()
