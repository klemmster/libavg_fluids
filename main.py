#!/usr/bin/env python
# encoding: utf-8

from libavg import AVGApp, avg

g_Player = avg.Player.get()

g_Size = (512, 512)

class ShadowDrawApp(AVGApp):
    def init(self):
        g_Player.loadPlugin("fluidnode")
        self.fluidsInstance = fluidnode.FluidNode(size=g_Size)
        self.fluidsInstance.registerInstance(self.fluidsInstance, self._parentNode)

ShadowDrawApp.start(resolution=g_Size)

