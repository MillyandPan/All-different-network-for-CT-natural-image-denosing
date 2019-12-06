thedir = '.\image\val_256_denoise_Gaussian_3\Noisy\Noisy';
% for ind = 0:15
%     image = imread([thedir,'00',num2str(ind),'.jpg']);
%     for cha = 1:3
%         K = wiener2(image(:,:,cha),[5 5]);
%         final_image(:,:,cha) = K;
%     end
%     imwrite(uint8(final_image),['.\image\val_256_denoise_Gaussian_3_CNN\Wiener00',num2str(ind),'.jpg']);
% end
listname = dir(['.\image\val_256_denoise_Gaussian_3_TWSC\Pre\*.jpg']);
MSE=0;
SSIM=0;
PSNR = 0;
for ind = 0:9
    image1 = imread(['.\image\val_256_denoise_Gaussian_3_TWSC\Truth\Truth','00',num2str(ind),'.jpg']);
    image2 = imread(['.\image\val_256_denoise_Gaussian_3_TWSC\Pre\Pre','00',num2str(ind),'.jpg']);
    image3 = imread(['.\image\val_256_denoise_Gaussian_3_TWSC\Noisy\Noisy','00',num2str(ind),'.jpg']);
    MSE = MSE+ immse(image1,image2);
    ssimval = ssim(image1,image2);
    SSIM = SSIM+ssimval;
    thepsnr=psnr(image1,image2);
    PSNR = PSNR+ thepsnr;
end
MSE = MSE/10; 
SSIM = SSIM/10;
PSNR = PSNR/10;