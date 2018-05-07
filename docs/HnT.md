----- .bg-white .slide-top
@unsplash(1-nx1QR5dTE)

.wrap
 @h1 .text-data **Corinthian**
 @h4 a photo filter for nightmares


@footer
 @div .wrap @div .span
  @button(href="https://github.com/thoppe/corinthian_filter") .alignleft .ghost
   ::github:: github.com/thoppe/corinthian_filter
  @button(href="https://twitter.com/metasemantic") .ghost .alignright
   ::twitter:: @metasemantic 

---- .align-left .bg-black
@unsplash(KSg_Uj5CM3Q) .dark
.wrap
	.text-landing left of the uncanny valley lies the
	.text-data Nope valley
----
@background(images/outside/neil-gaiman-credit-sasha-maslav_the-new-york-times_redux.jpg)
@h3
   Project <br> inspired <br> by <br> Neil Gaiman's <br> _Sandman_ <br> character, <br> Corinthian

----
.grid
    | @h1 **Isolate a face**
    | @figure(images/debug0.jpg)
----
.grid
    | @h1 **Find the eyes**
    | @figure(images/debug2.jpg)
----
.grid
    | @h1 **Find the mouth**	
    | @figure(images/debug1.jpg)
---- .wrap 
@h1 **Transform**	
.grid
    | @figure(images/outside/corinthian_comic.jpg)
    | @figure(images/debug3.jpg)
---- .wrap 
.text-landing @h2 Technical details
@h4
    + Everything is scripted
    + Faces are found with a CNN
    + Facial landmark masks from dlib
    + Mouths are placed on the eye center-of-mass
    + Masks expanded with a simple block convolution
    + Mini-mouths sized by sqrt of eye/face ratios
    + Sizing is clipped from 0.5 to 1.2     
    + After pasting, mouths are expanded and in-filled
---- .wrap
.text-landing @h2 Code
.text-landing only a few hundred magic lines
```
cfilter = np.ones((3,3))
mouth = convolve(mouth, cfilter).astype(np.bool)

# Fill the mouth in if it isn't too open
mouth = morph.binary_fill_holes(mouth)

whole_face_pts = np.vstack([L[k] for k in L])
mouth_pts = np.vstack([L[k] for k in mouth_keys])

nose_pts = np.vstack([L[k] for k in ['nose_tip','nose_bridge']])
face_mask = get_mask(whole_face_pts, height, width)

mouth_to_face_ratio = np.sqrt(bounding_box_area(mouth_pts) / bounding_box_area(whole_face_pts) )

# Clip the ratio so the mouth-eyes don't get too small
mouth_to_face_ratio = np.clip(mouth_to_face_ratio, 0.5, 1.2)
left_eye = get_mask(L['left_eye'], height, width)

E0 = copy_mask(img, left_eye, mouth, mouth_to_face_ratio)

# Inpaint around the eyes one out and one in from the outer edge
d = morph.binary_dilation(E0,iterations=1) & (~E0) #& (~nose_mask)
d = morph.binary_dilation(d,iterations=1)
img = inpaint.inpaint_biharmonic(img, d, multichannel=True)
img = np.clip((img*255).astype(np.uint8), 0, 255)
```
really, just go here [https://github.com/thoppe/corinthian_filter](https://github.com/thoppe/corinthian_filter)