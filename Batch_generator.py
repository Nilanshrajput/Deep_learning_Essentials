
#This genrates batches of specified sizes, functions uded in this can fe find in other files in repo,(augmentation,preprocess)
#It can be upadatesd for genrating  batches of test files to
def image_generator(input_ids, batch_size = 32,is_training=0):
  
  while True:
    batch_paths = np.random.choice(a= input_ids, size = batch_size)
    
    batch_input = []
    batch_output = []
    
    for input_id in batch_paths:
      output = cv2.imread(join(path_gt, input_id+'-gt.pbm'))
      input = cv2.imread(join(path_org, input_id+'-org.jpg'))
      
      input = preprocess_image(input) 
      output = preprocess_image(output) 
      if is_training:
        augmented = aug(image=input, mask=output)
        input=augmented['image']
        output=augmented['mask']

      
      batch_input += [input]
      batch_output += [output]
   
    batch_x = np.array(batch_input)
    batch_y = np.array(batch_output)
    
    
    
    yield (batch_x, batch_y)
      
    
      