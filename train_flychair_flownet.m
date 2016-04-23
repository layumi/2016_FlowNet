function train_flychair_flownet(varargin)
addpath('../MATLAB');
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./flying_chair_url.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.label);
imdb.images.set = ones(1,ss(2));
imdb.images.set(ceil(rand(1,ceil(ss(2)/20))*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = flownet_nocrop();
net.conserveMemory = true;
%net = load('data/48net-cifar-v3-custom-dither0.1-128-3hard/f48net-cpu.mat');


% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

opts.train.batchSize = 32;
%opts.train.numSubBatches = 1 ;
opts.train.continue = true; %before 16 only objective6 
opts.train.gpus = 2;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = '/home/zzd/matconvnet-fcn-master/data/flownet_nocrop' ; 
opts.train.learningRate = 1e-3;
opts.train.weightDecay = 0.0005;
opts.train.momentum = 0.9 ;
opts.train.numEpochs =  200;%48 24
opts.train.constraint = 100; %for rmsprop gradient clip
opts.train.down = 99999;% when to down gradient
opts.train.derOutputs = {'objective2',0.002, 'objective3',0.01,'objective4',0.05,'objective5',0.25,'objective6',1} ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

%record
if(~isdir(opts.expDir))
    mkdir(opts.expDir);
    copyfile('flownet_nocrop.m',opts.expDir);
end

% Call training function in MatConvNet
[net,info] = cnn_train_daga(net, imdb, @getBatch,opts) ;


% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
% --------------------------------------------------------------------
imlist = imdb.images.data(:,batch) ;
labellist = imdb.images.label(:,batch);
batch_size = numel(batch);
im = zeros(384,512,6,batch_size,'single');
label = zeros(384,512,2,batch_size);
training = opts.learningRate>0;
meanv = [97.0091/255;99.1555/255;97.4125/255];
meanv = reshape(meanv,[1 1 3]);

for i=1:batch_size
    p1 = imlist{i};
    p2 = strcat(imlist{i}(1:end-5),'2.ppm');
    im1 = bsxfun(@minus,im2single(imread(p1)),meanv);
    im2 = bsxfun(@minus,im2single(imread(p2)),meanv);
    im(:,:,:,i) = cat(3,im1,im2);
    label(:,:,:,i) = readFlowFile(labellist{i})./15;
end

if(rand>0.5)
    im = fliplr(im);
    label = fliplr(label);
    label(:,:,1,:) = -label(:,:,1,:);
end
if(rand>0.5)
    im = flipud(im);
    label = flipud(label);
    label(:,:,2,:) = -label(:,:,2,:);
end

if(training)
    [im ,label]=random_cut( im,label);
end
%add white noise
dither = rand(size(im),'single');
m = mean(mean(mean(dither)));
dither = bsxfun(@minus,dither,m);
im = im + 0*dither;
label2 = single(imresize(label,1/4));
label3 = single(imresize(label,1/8));
label4 = single(imresize(label,1/16));
label5 = single(imresize(label,1/32));
label6 = single(imresize(label,1/64));
%label2 = label2*15;
%label2 = flowToColor(label2(:,:,:,1));
%figure(2);imshow(label2);
inputs = {'input',gpuArray(im),'label2',gpuArray(label2),'label3',gpuArray(label3),...
    'label4',gpuArray(label4),'label5',gpuArray(label5),'label6',gpuArray(label6)};
