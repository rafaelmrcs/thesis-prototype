FROM node:20-alpine AS builder

WORKDIR /app

COPY Solar_Energy_Potential_Forecasting_Prototype/package*.json ./
RUN npm ci

COPY Solar_Energy_Potential_Forecasting_Prototype/ .

ARG VITE_BACKEND_URL
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

RUN npm run build

# ── Production stage: serve with nginx ──────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY Solar_Energy_Potential_Forecasting_Prototype/nginx/default.conf.template /etc/nginx/templates/default.conf.template

CMD ["sh", "-c", "PORT=${PORT:-80}; export PORT; envsubst '$PORT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
