FROM node:20-alpine AS builder

WORKDIR /app

COPY Solar_Energy_Potential_Forecasting_Prototype/package*.json ./
RUN npm install

COPY Solar_Energy_Potential_Forecasting_Prototype/ .

ARG VITE_BACKEND_URL
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

RUN npm run build

# ── Production stage: serve with nginx ──────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
