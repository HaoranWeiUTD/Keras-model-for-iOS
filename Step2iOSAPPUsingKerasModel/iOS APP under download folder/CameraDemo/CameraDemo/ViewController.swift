//
//  ViewController.swift
//  CameraDemo
//
//  Created by Sehgal, Abhishek on 5/3/18.
//  Copyright © 2018 Sehgal, Abhishek. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML
var decision1 = "A"
var decision2 = "B"
var decision3 = "C"
var SwitchState = 1
class ViewController: UIViewController, FrameExtractorDelegate {
    
    var frameExtractor: FrameExtractor!
    let openCVWrapper = OpenCVWrapper()
    //HERE IS THE PLACE TO DEFINE THE CLASSIFICATION MODEL
    //let model1 = myModel_TransferCNN0224mac()
    //let model2 = abModel_TransferCNN0224mac()
    var cgImage: CGImage!
    
    var frameCount = 0;
    var movingAverageBuffer = MovingAverageBuffer(period: 15)
    
    var nFramesPredicted = 0
    var startTime: DispatchTime!
    var endTime: DispatchTime!
    var elapsedTime: UInt64?

    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictLabel: UITextView!
    @IBAction func MV(_ sender: UISwitch) {
        SwitchState = (SwitchState+1)%2
    }
    
    
    func toggleTorch(on: Bool) {
        guard let device = AVCaptureDevice.default(for: AVMediaType.video)
            else {return}
        
        if device.hasTorch {
            do {
                try device.lockForConfiguration()
                
                if on == true {
                    device.torchMode = .on
                } else {
                    device.torchMode = .off
                }
                
                device.unlockForConfiguration()
            } catch {
                print("Torch could not be used")
            }
        } else {
            print("Torch is not available")
        }
    }
    
    
    let updateLabel = { (probs: Dictionary<String, Double>) -> (String) in
        var label: String = ""
        let probs_sorted = probs.sorted(by: {$0.value > $1.value})
        
        label = probs_sorted[0].key
        decision1 = decision2
        decision2 = decision3
        decision3 = label
        if SwitchState == 1{
        if decision3 == decision2{
            return decision3
        }else if decision3 == decision1{
            return decision3
        }else if decision2 == decision1{
            return decision1
        }else{
            return "Wait"
        }
        }else{
            return label
        }
    }
    
    let updateLabel2 = { (probs: MLMultiArray) -> (String) in
        var label: String = ""
        label = "Non Diabetic Probability\t" + String(format:"%.2f", probs[0]) + "\n Diabetic Probability\t" + String(format: "%.2f", probs[1])
        return label
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        toggleTorch(on: true)
        frameExtractor = FrameExtractor()
        frameExtractor.delegate = self
        startTime = DispatchTime.now()
        DispatchQueue.global(qos: .background).async {
            while(true){
                if(self.cgImage != nil){
                    self.predict()
                }
            }
        }

    }
    
    func captured(image: UIImage) {
            imageView.image = image
            cgImage = openCVWrapper.preprocessImage(image, image_size: 299)?.cgImage
        
    }
    
    func predict() {
            let start = DispatchTime.now()
            let pixelBuffer = pixelBufferFromImage(image: cgImage)
            let output1 = try? model1.prediction(image: pixelBuffer!)
            let output2 = try? model2.prediction(image: pixelBuffer!)
            let end = DispatchTime.now()
            debugPrint("Processing time",
                       Double(end.uptimeNanoseconds-start.uptimeNanoseconds)/1_000_000_000)
            debugPrint(output1!.output1["Not DR"]!)
            DispatchQueue.main.async {
                if output1!.output1["Not DR"]! > 0.5{
                    debugPrint("work test")
                    self.predictLabel.text = self.updateLabel((output2?.output1)!)
                }
                else{
                    self.predictLabel.text = self.updateLabel((output1?.output1)!)
                }
            }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    
    func pixelBufferFromImage(image: CGImage) -> CVPixelBuffer? {
        let frameSize = CGSize(width: image.width, height: image.height)
        var pixelBuffer: CVPixelBuffer? = nil
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(frameSize.width),
                                         Int(frameSize.height),
                                         kCVPixelFormatType_32BGRA,
                                         nil,
                                         &pixelBuffer)
        if(status != kCVReturnSuccess) { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let data = CVPixelBufferGetBaseAddress(pixelBuffer!)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: data,
                                width: Int(frameSize.width),
                                height: Int(frameSize.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                                space: rgbColorSpace,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).rawValue | CGBitmapInfo(rawValue: CGImageByteOrderInfo.order32Little.rawValue).rawValue)
        context?.draw(image, in: CGRect(x: 0, y: 0, width: frameSize.width, height: frameSize.height))
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }


}

