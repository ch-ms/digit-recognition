pkg load image;
load 'learned.mat';
arg_list = argv ();
fname = arg_list{1};

imgBW = imread(strcat("./in/", fname), "jpg");
% size(imgRGB)
% imshow(imgRGB);
% anykey
% imgYIQ = rgb2ntsc(Image3DmatrixRGB);
% imgBW  = imgYIQ(:, :, 1);

% Scaling
s = size(imgBW, 1);
index = 1:(s / 20):s;
newImage = imgBW(index,index);

% Invert
iMat = im2double(newImage);
iMat = -iMat + 1;

% Gray & white

imshow(flipud(iMat), [-1 1]);

exmpl = iMat(:)';
size(exmpl)
pred = predict(Theta1, Theta2, exmpl);
fprintf("Predicted %i\n", pred);
anykey;
