FROM nginx:alpine

# Copy default nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static frontend HTML
COPY main.html /usr/share/nginx/html/main.html

# Expose port 80
EXPOSE 80
