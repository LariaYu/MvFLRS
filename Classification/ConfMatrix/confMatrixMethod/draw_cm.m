function draw_cm(mat,tick,num_class)
%%
%  Matlab code for visualization of confusion matrix;
%  Parameters£ºmat: confusion matrix;
%              tick: name of each class, e.g. 'class_1' 'class_2'...
%              num_class: number of class
%
%  Author£º Page( Ø§×Ó)  
%           Blog: www.shamoxia.com;  
%           QQ:379115886;  
%           Email: peegeelee@gmail.com
%%
imagesc(mat);            %# in color
colormap(flipud(gray));  %# for gray; black for large value.
set(gca,'xtick',1:1:num_class);  
set(gca,'ytick',1:1:num_class);  

% % textStrings = num2str(mat(:),'%0.2f'); 
% num=length(textStrings);
% for i=1:num
%     if textStrings(i)==0.00
%         textStrings(i)=0;
%     end
% end
textStrings =num2str(roundn(mat(:),-2));
textStrings = strtrim(cellstr(textStrings)); 
num=length(textStrings);
for i=1:num
    if textStrings{i}=='0'
        textStrings{i}='';
    end
end
[x,y] = meshgrid(1:num_class); 
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center','FontSize',8);
midValue = mean(get(gca,'CLim')); 
textColors = repmat(mat(:) > midValue,1,3); 
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

%   Examples:
%   >> bar( hsv(5)+0.05 )
%   >> days = {'Monday','Tuesday','Wednesday','Thursday','Friday'};
%   >> set( gca(), 'XTickLabel', days )
%   >> rotateXLabels( gca(), 45 )


set(gca,'xticklabel',tick,'XAxisLocation','top');

% rotateXLabels(gca(), 315);% rotate the x tick

set(gca,'yticklabel',tick);


