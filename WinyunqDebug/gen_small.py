from PIL import Image
img = Image.new('RGB', (128, 128), color = 'red')
img.save('small_test.png')
print("Successfully generated small_test.png!")
