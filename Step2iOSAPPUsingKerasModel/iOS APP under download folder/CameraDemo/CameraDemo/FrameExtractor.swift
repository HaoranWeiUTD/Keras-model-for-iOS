//
//  FrameExtractor.swift
//  CameraDemo
//
//  Created by Sehgal, Abhishek on 5/3/18.
//  Copyright © 2018 Sehgal, Abhishek. All rights reserved.
//

import UIKit
import AVFoundation

protocol FrameExtractorDelegate: class {
    func captured(image: UIImage)
}

/*
 This file needs to do three things:
 1. It should access the camera
 2. It should be customizable (front/back, orientation, quality)
 3. It should return every frame captured
 */
class FrameExtractor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session queue")
    private let position = AVCaptureDevice.Position.back
    private let quality = AVCaptureSession.Preset.cif352x288
    private let context = CIContext()
    
    weak var delegate: FrameExtractorDelegate?
    
    override init() {
        super.init()
        sessionQueue.async { [unowned self] in
            self.configureSession()
            self.captureSession.startRunning()
        }
    }
    
    private func configureSession() {
        captureSession.sessionPreset = quality
        guard let captureDevice = selectCaptureDevice() else {return}
        guard let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice) else {return}
        guard captureSession.canAddInput(captureDeviceInput) else {return}
        captureSession.addInput(captureDeviceInput)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sample buffer"))
        
        guard captureSession.canAddOutput(videoOutput) else {return}
        captureSession.addOutput(videoOutput)
        
        guard let connection = videoOutput.connection(with: AVMediaType.video) else {return}
        guard connection.isVideoOrientationSupported else {return}
        guard connection.isVideoMirroringSupported else {return}
        connection.videoOrientation = .portrait
        connection.isVideoMirrored = position == .front
        
    }
    
    private func selectCaptureDevice() -> AVCaptureDevice? {
        return AVCaptureDevice.devices().filter {
            ($0 as AnyObject).hasMediaType(AVMediaType.video) &&
            ($0 as AnyObject).position == position
            }.first
    }
    
    private func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return nil}
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {return nil}
        return UIImage(cgImage: cgImage)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let uiImage = self.imageFromSampleBuffer(sampleBuffer: sampleBuffer) else {return}
        DispatchQueue.main.async {[unowned self] in
            self.delegate?.captured(image: uiImage)
        }
    }
}
