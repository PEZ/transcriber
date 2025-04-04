(ns pez.transcribe
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [com.phronemophobic.whisper :as whisper]))

(comment
  ;; First init the model - using load-model instead of init
  (def model (whisper/load-model "models/ggml-model-f32"))

  ;; Then use the initialized model for recording
  (def get-text (whisper/record-and-transcribe model))

  ;; Get the transcription when ready
  (def transcription (get-text)))

