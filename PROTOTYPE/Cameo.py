import cv2
from WindowManager import WindowManager
from CaptureManager import CaptureManger
from laneline_detection import Lane_Lines_Video


class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManger(cv2.VideoCapture(0), self._windowManager, False)

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = Lane_Lines_Video.process_image(frame)
            self._captureManager.frame = frame

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress

        space -> Take a screenshot
        tab -> Start/stop recoding a screencast
        escape -> Quit.
        """
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
