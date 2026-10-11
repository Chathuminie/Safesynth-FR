const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");

const ctx = canvas.getContext("2d");
const overlayCtx = overlay.getContext("2d");

const startCam = document.getElementById("startCam");
const stopCam = document.getElementById("stopCam");
const startLive = document.getElementById("startLive");
const stopLive = document.getElementById("stopLive");

const label = document.getElementById("label");
const conf = document.getElementById("conf");
const progressFill = document.getElementById("progressFill");
const statusText = document.getElementById("detectionStatus");
const note = document.getElementById("note");
const camStatus = document.getElementById("camStatus");

const intervalInput = document.getElementById("intervalMs");

let stream = null;
let detectTimer = null;

function setCameraError(message) {
camStatus.textContent = message
statusText.textContent = "Webcam unavailable"
note.textContent = message
label.textContent = "-"
conf.textContent = "-"
progressFill.style.width = "0%"
overlayCtx.clearRect(0,0,overlay.width,overlay.height)
startCam.disabled = false
stopCam.disabled = true
startLive.disabled = true
stopLive.disabled = true
}

function getCameraErrorMessage(err) {
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
return "Webcam unavailable. Your browser does not support camera access."
}

if (!window.isSecureContext) {
return "Webcam unavailable. Mobile browsers require HTTPS or localhost for camera access."
}

if (!err || !err.name) {
return "Webcam unavailable. Please check your camera and try again."
}

if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
return "Webcam unavailable. No camera device was found."
}

if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
return "Webcam unavailable. Camera permission was denied."
}

if (err.name === "NotReadableError" || err.name === "TrackStartError") {
return "Webcam unavailable. The camera is already in use by another app."
}

if (err.name === "OverconstrainedError" || err.name === "ConstraintNotSatisfiedError") {
return "Webcam unavailable. Camera requirements could not be satisfied."
}

return "Webcam unavailable. Please check your camera and try again."
}


function drawOverlay(person, confidence) {

overlayCtx.clearRect(0,0,overlay.width,overlay.height)

const x = overlay.width * 0.25
const y = overlay.height * 0.15
const w = overlay.width * 0.5
const h = overlay.height * 0.65

overlayCtx.lineWidth = 4
overlayCtx.strokeStyle = "#00ff88"
overlayCtx.strokeRect(x,y,w,h)

overlayCtx.fillStyle = "#00ff88"
overlayCtx.font = "18px Arial"

overlayCtx.fillText(person + " (" + confidence + "%)", x, y - 10)

}


function updateResult(data){

if(!data.ok){
statusText.textContent="Error"
return
}

label.textContent=data.label

let confidence=Math.round(data.confidence*100)

statusText.textContent=data.detection_status

if(data.label==="No Face"){
conf.textContent="-"
progressFill.style.width="0%"
progressFill.style.background="#ef4444"
note.textContent="No face detected in the current frame"
overlayCtx.clearRect(0,0,overlay.width,overlay.height)
return
}

conf.textContent=confidence+"%"

progressFill.style.width=confidence+"%"

if(data.label==="Unknown"){
note.textContent=data.detection_status==="multiple_faces"
? "Multiple faces detected. The largest face was analyzed but not recognized."
: "Face detected but not recognized"
progressFill.style.background="#ff9800"
drawOverlay("Unknown",confidence)
}else if(data.label==="Uncertain"){
note.textContent=data.detection_status==="multiple_faces"
? "Multiple faces detected. The largest face has a possible match, but the result is uncertain."
: "Possible match found, but the result is uncertain."
progressFill.style.background="#ff9800"
drawOverlay("Uncertain",confidence)
}else if(data.detection_status==="multiple_faces"){
note.textContent="Multiple faces detected. The largest face was analyzed."
progressFill.style.background="#ff9800"
drawOverlay(data.label,confidence)
}else{
note.textContent="Identity recognized successfully"
progressFill.style.background="#4caf50"
drawOverlay(data.label,confidence)
}

}


function captureFrame(){

if(!stream) return

canvas.width = video.videoWidth
canvas.height = video.videoHeight
overlay.width = video.videoWidth
overlay.height = video.videoHeight

ctx.drawImage(video,0,0,canvas.width,canvas.height)

canvas.toBlob(async function(blob){

const formData=new FormData()
formData.append("image",blob,"frame.jpg")

try{

const res=await fetch("/predict_webcam",{method:"POST",body:formData})

const data=await res.json()

updateResult(data)

}catch(err){

statusText.textContent="Server error"
note.textContent="Connection failed"

}

},"image/jpeg")

}


startCam.onclick=async()=>{

try{
if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
setCameraError("Webcam unavailable. Your browser does not support camera access.")
return
}

if(!window.isSecureContext){
setCameraError("Webcam unavailable. Open this app over HTTPS on mobile, such as an ngrok link.")
return
}

stream=await navigator.mediaDevices.getUserMedia({
video:{
facingMode:"user",
width:{ideal:1280},
height:{ideal:720}
}
})

video.srcObject=stream
await video.play()

startCam.disabled=true
stopCam.disabled=false
startLive.disabled=false

camStatus.textContent="Camera running"
statusText.textContent="Camera ready"
note.textContent="Start recognition when you are ready."

}catch(err){

setCameraError(getCameraErrorMessage(err))

}

}


stopCam.onclick=()=>{

if(stream){
stream.getTracks().forEach(track=>track.stop())
stream=null
}

video.srcObject=null

clearInterval(detectTimer)

overlayCtx.clearRect(0,0,overlay.width,overlay.height)

startCam.disabled=false
stopCam.disabled=true
startLive.disabled=true
stopLive.disabled=true

camStatus.textContent="Camera stopped"
statusText.textContent="Waiting for camera"
note.textContent="Start the camera to begin recognition."

}


startLive.onclick=()=>{

const interval=parseInt(intervalInput.value)||700

detectTimer=setInterval(captureFrame,interval)

startLive.disabled=true
stopLive.disabled=false

statusText.textContent="Live detection running"

}


stopLive.onclick=()=>{

clearInterval(detectTimer)

overlayCtx.clearRect(0,0,overlay.width,overlay.height)

startLive.disabled=false
stopLive.disabled=true

statusText.textContent="Detection paused"

}
