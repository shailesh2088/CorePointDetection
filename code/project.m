%--------------------------------------------------------------------------
%Shailesh Samudrala
%Core Point Detection In Fingerprint Images
%Project - EEL6562
%--------------------------------------------------------------------------

%Step 1 - Image segmentation

img = imread ('fingerprint1.tif');
[M,N] = size(img);             
img = double(img);

mean = zeros(M/4,N/4);
variance = zeros(M/4,N/4);
%x,y - indices for mean and variance
x =0;
y =0;
sum = 0;
var =0;
%threshold for segmentation
threshold = 0.1;

%creating non-overlapping blocks of 4x4
for i = 1:4:M-3,        
    x = x+1;
    y=0;
    for j = 1:4:N-3,
        y = y+1;
        sum =0;
        var =0;
        %inner loop to calculate mean
        for k = i:i+3,
            for l = j:j+3,
                sum = sum + img(k,l);
            end
        end
        mean(x,y) = sum/16;
        %inner loop to calculate variance
        for k = i:i+3,
            for l = j:j+3,
                var = (img(k,l) - mean(x,y))^2;
            end
        end
        variance(x,y) = var/16;    
    end
end

%create intermediate output, to be further cropped
out = ones(M,N);
out = 255*out;
out = double(out);
x =0;
y =0;
for i = 1:4:M-3,
    x = x+1;
    y =0;
    for j = 1:4:N-3,
        y = y+1;
        for k = i:i+3,
            for l = j:j+3,
                 if(variance(x,y)>threshold),
                    out(k,l) = img(k,l);
                 end
            end
        end
    end
end


%find minimum Y-coordinate to be cropped

a_min =10000;
x =1;
y =1;
for i = 1 : M,
    a=10000;
    y=1;
    for j = 1:N,
        if(variance(x,y) > threshold),
            a = j;
            break;
        end
        if(mod(j,4)==0),
            y = y+1;
        end
    end
    if(a_min>a),
    a_min = a;
    end
    if(mod(i,4)==0),
    x = x+1;
    end
end      


%find maximum Y-coordinate to be cropped
a =0;
a_max =0;
x =1;
y =N/4+1;
for i = 1 :M,
    a=0;
    y=N/4;
    for j = N-1:-1:1,
        if(variance(x,y) > threshold),
            a = j;
           break;
        end
        if(mod(j,4)==0),
            y = y-1;
        end
    end
    if(a>a_max),
    a_max = a;
    end
    if(mod(i,4)==0),
    x = x+1;
    end
end      

%find minimum X-coordinate to be cropped
b_min =10000;
x =1;
y =1;
for j = 1 : N,
    b=10000;
    x=1;
    for i = 1:M,
        if(variance(x,y) > threshold),
            b= i;
            break;
        end
        if(mod(i,4)==0),
            x = x+1;
        end
    end
    if(b_min>b),
    b_min = b;
    end
    if(mod(j,4)==0),
    y = y+1;
    end
end      

%find maxmimum X-coordinate to be cropped
b_max =0;
x =1;
y =1;
for j = 1 : N,
    b=0;
    x=M/4;
    for i = M-1:-1:1,
       if(variance(x,y) > threshold),
            b= i;
            break;
        end
        if(mod(i,4)==0),
            x = x-1;
        end
    end
    if(b>b_max),
    b_max = b;
    end
    if(mod(j,4)==0),
    y = y+1;
    end
end      


%crop image to remove background
x=0;
y=0;
final_out = zeros(b_max-b_min,a_max-a_min);
for i = b_min:b_max,
    x=x+1;
    y =0;
    for j = a_min:a_max,
        y=y+1;
        final_out(x,y) = out(i,j);
    end
end

final_out = uint8(final_out);
out = uint8(out);
figure,imshow(final_out);
imwrite(final_out,'seg_out.tif');   


%-----------------END OF STEP 1 -------------------------------------------

%Step 2 - Normalization

img = imread('fingerprint1.tif');
[M,N] = size(img);
img = double (img);


%find sum of intensity
intensity = 0;
for i = 1:M,
    for j = 1:N,
        intensity = intensity + img(i,j);
    end
end


%mean of image intensity
mean = intensity/(M*N);


%find variance of image intensity
var = 0;
for i = 1:M,
    for j =1:N,
        var = (img(i,j) - mean)^2;
    end
end

variance = var/(M*N);


%desired mean and variance values
M0 = 100;
V0 = 100;

%normalized output matrix
normalization = zeros(M,N);
normalization = double(normalization);


%find normalized values for each pixel
for i = 1:M,
    for j = 1:N,
        if(img(i,j)>mean)
            normalization(i,j) = M0 + sqrt(V0*(img(i,j)-mean)^2/variance);
        else
            normalization(i,j) = M0 - sqrt(V0*(img(i,j)-mean)^2/variance);
        end
    end
end

img = uint8(img);
normalization = uint8(normalization);


%figure,imshow(normalization);
imwrite(normalization,'normalization.tif');

%------------------END OF STEP 2 ------------------------------------------



%Step 3 - Orientation Field Estimation

img = imread('normalization.tif');
[M,N] = size(img);
img = double(img);

%block size
w = 3;
height = floor(M/w);
width = floor(N/w);


orientation = zeros(height,width);
orientation = double(orientation);

%gradient calculation

grad_x = ones(M,N);
grad_x = double(grad_x);
grad_y = ones(M,N);
grad_y = double(grad_y);

for i = 1:M-2,
    for j = 1:N-2,
         gx = (-1*img(i,j)+ -2*img(i,j+1)+ -1*img(i,j+2)+ img(i+2,j)+ 2*img(i+2,j+1)+ img(i+2,j+2))/4;
         gy = (-1*img(i,j) + img(i,j+2)+ -2*img(i+1,j) + 2*img(i+1,j+2)+ -1*img(i+2,j) + img(i+2,j+2))/4;
         grad_x(i,j) = gx;
         grad_y(i,j) = gy;
    end
end


%Vx and Vy are local field orientations
vx = zeros(height,width);
vx = double(vx);
vy = zeros(height,width);
vy = double(vy);

x =1;
y =1;
for i = 1:M-1,
    y =1;
    for j = 1:N-1,
         vx (x,y) = vx(x,y)+ 2*grad_x(i,j)*grad_y(i,j);
         vy (x,y) = vy(x,y)+ grad_x(i,j)^2*grad_y(i,j)^2;
         if(mod(j,w) == 0),
             y = y+1;
         end
    end
    if(mod(i,w) == 0),
        x = x+1;
    end
end


%orientation field estimation
for i = 1:height,
    for j =1:width,
        if(vx(i,j) == 0),
            orientation(i,j) = 0;
        else
            orientation(i,j) = 0.5*atan(vy(i,j)/vx(i,j));
        end
    end
end

% continuous vector field

phi_x = zeros(height,width);
phi_y = zeros(height,width);
phi_x = double(phi_x);
phi_y = double(phi_y);
for i = 1:height,
    for j = 1:width,
        phi_x(i,j) = cos(2*orientation(i,j));
        phi_y(i,j) = sin(2*orientation(i,j));
    end
end

%low pass averaging filter
h = ones(3,3);
h = h/9;

field_x = filter2(h,phi_x);
field_y = filter2(h,phi_y);


%smoothed orientation field
smoothed = zeros(height,width);
smoothed = double(smoothed);

for i = 1:height,
    for j = 1:width,
        smoothed(i,j) = 0.5*atan(field_y(i,j)/field_x(i,j));
    end
end

%------------------END OF STEP 3 ------------------------------------------
 


%Step 4 - Detection of Curvature & Core point detection


%window size
w = 3;

x =1;
y =1;
Diff_Y = 0;
Diff_X = 0;
Diff_Y = double(Diff_Y);
Diff_X = double(Diff_X);

min_dx = 0;
min_dy = 0;
A = 80;
B = 100;
C = 65;
D = 81;
mean_i =0;
mean_j =0;
count_i  =0;
count_j = 0;

for i = 1:w:height-2,
    Diff_Y = 0;
    Diff_X = 0;
    for j = 1:w:width-2,
         Diff_Y = Diff_Y + (sin(2*smoothed(i,j+2)) + sin(2*smoothed(i+1,j+2)) + sin(2*smoothed(i+2,j+2)) - sin(2*smoothed(i,j)) - sin(2*smoothed(i+1,j)) - sin(2*smoothed(i+2,j)));
         Diff_X = Diff_X + (cos(2*smoothed(i+2,j)) + cos(2*smoothed(i+2,j+1)) + cos(2*smoothed(i+2,j+2)) - cos(2*smoothed(i,j)) - cos(2*smoothed(i,j+1)) - cos(2*smoothed(i,j+2)));
                             
         if(Diff_X < 0),
             if(Diff_Y <0),
                 if(A<i),
                     if(i<B),
                          mean_i = mean_i + i;
                          count_i = count_i+1;
                     end
                 end
                 if(C<j),
                     if(j<D),
                         mean_j = mean_j + j;
                         count_j = count_j+1;
                     end
                 end
                  
             end
         end
    end
end

block_i = mean_i/count_i;
block_j = mean_j/count_j;
    
%core point coordinates
core_x = floor(block_i)*w -1
core_y = floor(block_j)*w -1


%---------------END OF STEP 4----------------------------------------------


% Step 5 - Identify ROI


img=imread('fingerprint1.tif');
img = double(img);
x =0;
y =0;

%Region Of Interest Matrix
ROI = zeros(100,100);
ROI = double(ROI);

for i = core_x -50: core_x + 50,
    x = x+1;
    y = 0;
    for j = core_y -50: core_y + 50,
        y = y+1;
        ROI(x,y) = img(i,j);
    end
end


img = uint8(img);
ROI=uint8(ROI);
figure,imshow(ROI);
figure,imshow(img);
imwrite(ROI,'ROI.tif');

        
%----------------END OF STEP 5 --------------------------------------------